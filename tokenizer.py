import os

from abc import ABC, abstractmethod

from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, TypeVar

import regex as re
from sortedcontainers import SortedSet

from cs336_basics.pretokenization_example import find_chunk_boundaries

@dataclass
class BPETokenizer:
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]


@dataclass
class TokenSplit:
    id_to_token: List[bytes]
    tokens: List[List[int]]


def split_text_into_docs(all_docs: str, special_tokens: Set[str]) -> List[str]:
    special_token_regex = re.compile("|".join(special_tokens))
    # XXX this splits on all special tokens; do we want to separate docs only on eod?
    return re.split(special_token_regex, all_docs)


class TokenPresplitter(ABC):
    def __init__(self, special_tokens: Set[str]):
        self.token_lookup: Dict[str, int] = {}
        self.tokens: List[List[int]] = []
        self.special_tokens: Set[str] = special_tokens


    def _presplit_tokens(self, text: str) -> None:
        """
            each class instance builds its own indepdendent token id mapping to allow for pure parallelization without
            sharing data across threads.  they will be merged back together after each thread finishes

            XXX: given that token mappings should be extremely read-heavy, is it actually faster to just use a shared
            mapping and eat the cross-thread contention overhead?
        """

        tokens: List[int] = []

        TOKEN_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for match in re.finditer(TOKEN_REGEX, text):
            s = match.group(0)
            if not s:
                continue

            token = self.token_lookup.setdefault(s, len(self.token_lookup))
            tokens.append(token)

        self.tokens.append(tokens)


    @abstractmethod
    def _get_docs(self) -> Iterable[str]:
        pass

    def process(self) -> None:
        for doc in self._get_docs():
            self._presplit_tokens(doc)

    def get_token_split(self) -> TokenSplit:
        assert list(sorted(self.token_lookup.values())) == list(range(len(self.token_lookup)))

        id_to_token = [bytes()] * len(self.token_lookup)
        for token_bytes, token_id in self.token_lookup.items():
            id_to_token[token_id] = bytes(token_bytes, encoding="utf-8")

        return TokenSplit(id_to_token=id_to_token, tokens=self.tokens)


class FileChunkPresplitter(TokenPresplitter):
    def __init__(self, special_tokens: Set[str], data_path: os.PathLike, start_index: int, end_index: int):
        super().__init__(special_tokens)
        self.data_path = data_path
        self.start_index = start_index
        self.end_index = end_index


    def _get_docs(self) -> Iterable[str]:
        with open(self.data_path) as input_file:
            input_file.seek(self.start_index)
            input_data = input_file.read(self.end_index - self.start_index)
        
        for doc in split_text_into_docs(input_data, self.special_tokens):
            yield doc

        # return [input_data]


class StringDocPresplitter(TokenPresplitter):
    def __init__(self, special_tokens: Set[str], docs: List[str]):
        super().__init__(special_tokens)
        self.docs = docs


    def _get_docs(self) -> Iterable[str]:
        return self.docs


class TokenMerger:
    def __init__(self, old_pair: Tuple[int, int], new_token: int, inplace: bool = True) -> None:
        self.old_pair = old_pair
        self.new_token = new_token
        self.inplace = inplace

        self.deltas: DefaultDict[Tuple[int, int], int] = defaultdict(int)
        self.updated_docs: List[List[int]] = []


    def merge_tokens(self, doc: List[int]) -> None:
        """
            !!! modifies doc in-place, replacing all instances of old_pair with new_token

            returns deltas of token pair counts to be consolidated by central coordinating thread

            factored into separate function to make more conducive to parallelizing, replacing with rust implementation, etc.
            (or even rpc call to distribute across cluster)
        """
        write_cursor: int = 0
        read_cursor: int = 0

        if not self.inplace:
            doc = list(doc)

        while read_cursor < len(doc):
            if read_cursor < len(doc) - 1 and tuple(doc[read_cursor:read_cursor + 2]) == self.old_pair:
                self.deltas[(doc[read_cursor], doc[read_cursor + 1])] -= 1

                # decrement counts of pairs that were previously formed on either side
                # and replace them with newly formed pairs from substituting the new token
                if read_cursor > 0:
                    # when removing the trailing token pair, we need to look behind the write cursor rather than the read cursor,
                    # because we may have already replaced the previous token with a substitution
                    assert write_cursor > 0 # must be > 0 because we advance it at least one for each read step

                    self.deltas[(doc[write_cursor - 1], doc[read_cursor])] -= 1
                    self.deltas[(doc[write_cursor - 1], self.new_token)] += 1

                if read_cursor < len(doc) - 2:
                    self.deltas[(doc[read_cursor + 1], doc[read_cursor + 2])] -= 1
                    self.deltas[(self.new_token, doc[read_cursor + 2])] += 1

                doc[write_cursor] = self.new_token
                read_cursor += 2
            else:
                doc[write_cursor] = doc[read_cursor]
                read_cursor += 1

            # we are writing one new token regardless
            write_cursor += 1
        
        # truncate document based on how many tokens we wrote
        doc[write_cursor:] = []
        self.updated_docs.append(doc)
        

def merge_tokens(docs: List[List[int]], old_pair: Tuple[int, int], new_token: int) -> Dict[Tuple[int, int], int]:
    merger = TokenMerger(old_pair, new_token)
    for doc in docs:
        merger.merge_tokens(doc)
    return merger.deltas


# have to wrap this in a basic function to allow for pickling in multiprocessing pool
def presplit_docs(presplitter: TokenPresplitter) -> TokenSplit:
    presplitter.process()
    return presplitter.get_token_split()


T = TypeVar("T")
def _chunkify(l: List[T], chunk_size: int) -> List[List[T]]:
    i: int = 0
    chunks: List[List[T]] = []
    while i < len(l):
        chunks.append(l[i : i + chunk_size])
        i += chunk_size
    return chunks


class TokenizerTrainer:
    def __init__(self, vocab_size: int, special_tokens: Iterable[str]):
        self.vocab_size: int = vocab_size
        self.special_tokens: Set[str] = set(special_tokens)
        self._presplit_sentinel: int = self.vocab_size + 1


    def train_on_file(self, path: os.PathLike, n_workers: Optional[int] = None):
        n_workers = n_workers or os.cpu_count()
        assert n_workers is not None

        with open(path, "rb") as input_file:
            chunk_boundaries = find_chunk_boundaries(input_file, n_workers, b"<|endoftext|>")

        chunks: Iterable[Tuple[int, int]] = zip(chunk_boundaries[:-1], chunk_boundaries[1:])
        
        pool = Pool(n_workers)
        token_presplits = pool.map(presplit_docs, [FileChunkPresplitter(self.special_tokens, path, *chunk) for chunk in chunks])

        return self._merge_presplits(token_presplits, n_workers)

    def train_on_strings(self, docs: List[str], n_workers: Optional[int] = None):
        n_workers = n_workers or os.cpu_count()
        assert n_workers is not None

        pool = Pool(n_workers)
        token_presplits = pool.map(presplit_docs, [StringDocPresplitter(self.special_tokens, chunk) for chunk in _chunkify(docs, 100)])

        return self._merge_presplits(token_presplits, n_workers)

    def _merge_presplits(self, token_presplits: List[TokenSplit], n_workers: int) -> BPETokenizer:
        thread_pool = ThreadPool(n_workers)
        
        vocab: List[bytes] = [bytes([b]) for b in range(256)]
        token_lookup: Dict[bytes, int] = { token: i for i, token in enumerate(vocab) }

        def _add_token(token: bytes):
            assert token not in token_lookup
            token_lookup[token] = len(vocab)
            vocab.append(token)

        merges: List[Tuple[bytes, bytes]] = []
        doc_tokens: List[List[int]] = []
        byte_pair_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)

        # per spec, sort lexicographically on token contents in case of count ties
        def pair_sort_key(x: Tuple[int, Tuple[int, int]]) -> Tuple[int, bytes, bytes]:
            return (x[0], vocab[x[1][0]], vocab[x[1][1]])

        top_counts: SortedSet[Tuple[int, Tuple[int, int]]] = SortedSet(key=pair_sort_key) # type: ignore

        for token_split in token_presplits:
            for doc in token_split.tokens:
                single_doc_tokens: List[int] = []
                for batch_token_id in doc:
                    token_value: bytes = token_split.id_to_token[batch_token_id]
                    single_doc_tokens.extend(list(token_value))
                    single_doc_tokens.append(self._presplit_sentinel)

                    # XXX could also track counts in pre-split to make this faster
                    for i in range(len(token_value) - 1):
                        byte_pair_counts[tuple(token_value[i:i+2])] += 1

                doc_tokens.append(single_doc_tokens)

        # initialize byte pair ordering
        for pair, top_count in byte_pair_counts.items():
            top_counts.add((top_count, pair))

        while len(vocab) < self.vocab_size - len(self.special_tokens):
            if len(top_counts) == 0:
                print(f"exhausted all token pairs with vocab size of only {len(vocab)}")
                break
            
            top_count, top_pair = next(reversed(top_counts))
            t1, t2 = top_pair

            # register new token
            merges.append((vocab[t1], vocab[t2]))
            new_token: bytes = vocab[t1] + vocab[t2]
            _add_token(new_token)

            # print(f"replacing {top_count} instances of pair {top_pair}: {(vocab[t1], vocab[t2])} with new token {new_token} ({len(vocab) - 1})")
            # print(f"created new token {new_token} with id {token_lookup[new_token]}")

            # merge token pair in each document and update pair counts accordingly
            # note: this modifies `doc` in place for efficiency
            #merge_results = thread_pool.map(lambda docs: merge_tokens(docs, top_pair, token_lookup[new_token]), _chunkify(doc_tokens, 10))
            merge_results = [merge_tokens(docs, top_pair, token_lookup[new_token]) for docs in _chunkify(doc_tokens, 10)]
            for pair_deltas in merge_results:
                for changed_pair, delta in pair_deltas.items():
                    if self._presplit_sentinel in changed_pair:
                        continue

                    #print(f"changing count of {changed_pair} by {delta}: {byte_pair_counts.get(changed_pair,0)} + {delta}")
                    if byte_pair_counts.get(changed_pair, 0):
                        top_counts.remove((byte_pair_counts[changed_pair], changed_pair))
                    byte_pair_counts[changed_pair] += delta
                    if byte_pair_counts[changed_pair] > 0:
                        top_counts.add((byte_pair_counts[changed_pair], changed_pair))

            # we should have removed all entries naturally as part of processing above
            for count, pair in top_counts:
                if pair == top_pair:
                    for doc in doc_tokens:
                        print("\n".join(str(vocab[token] if token != self._presplit_sentinel else "") for token in doc))
                    print(f"still {count} pairs remaining of processed pair {top_pair}")
                    assert False

        for special_token in self.special_tokens:
            _add_token(bytes(special_token, encoding="utf-8"))

        return BPETokenizer(vocab=dict(enumerate(vocab)), merges=merges)


if __name__ == "__main__":
    from scalene import scalene_profiler

    eod_token = "<|endoftext|>"

    scalene_profiler.start()
    with scalene_profiler.enable_profiling():
        t = TokenizerTrainer(10_000, {eod_token}).train_on_file("/Users/brendon/src/cs336-a1/TinyStoriesV2-GPT4-train.txt")
    scalene_profiler.stop()
