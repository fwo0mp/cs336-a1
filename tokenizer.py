from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from typing import DefaultDict, Dict, Generic, List, Optional, Set, Tuple, TypeVar

import regex as re
from sortedcontainers import SortedSet

@dataclass
class BPETokenizer:
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]


@dataclass
class TokenSplit:
    id_to_token: List[bytes]
    tokens: List[List[int]]


class TokenPresplitter:
    def __init__(self):
        self.token_lookup: Dict[bytes, int] = {}
        self.tokens: List[List[int]] = []

    def _presplit_tokens(self, text: bytes) -> None:
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

            token = self.token_lookup.setdefault(bytes(s, encoding="utf-8"), len(self.token_lookup))
            tokens.append(token)

        self.tokens.append(tokens)

    
    def process(self, docs: List[bytes]) -> TokenSplit:
        for doc in docs:
            self._presplit_tokens(doc)

        assert list(sorted(self.token_lookup.values())) == list(range(len(self.token_lookup)))

        id_to_token = [bytes()] * len(self.token_lookup)
        for token_bytes, token_id in self.token_lookup.items():
            id_to_token[token_id] = token_bytes

        return TokenSplit(id_to_token=id_to_token, tokens=self.tokens)


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
                self.deltas[tuple(doc[read_cursor:read_cursor + 2])] -= 1

                # decrement counts of pairs that were previously formed on either side
                # and replace them with newly formed pairs from substituting the new token
                if read_cursor > 0:
                    self.deltas[tuple(doc[read_cursor - 1:read_cursor + 1])] -= 1
                    self.deltas[(doc[read_cursor - 1], self.new_token)] += 1
                if read_cursor < len(doc) - 2:
                    self.deltas[tuple(doc[read_cursor + 1:read_cursor + 3])] -= 1
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


def presplit_docs(docs: List[bytes]) -> TokenSplit:
    return TokenPresplitter().process(docs)


T = TypeVar("T")
def _chunkify(l: List[T], chunk_size: int) -> List[List[T]]:
    i: int = 0
    chunks: List[List[T]] = []
    while i < len(l):
        chunks.append(l[i : i + chunk_size])
        i += chunk_size
    return chunks


def train_tokenizer(docs: List[bytes], vocab_size: int, special_tokens: Set[str], n_workers: Optional[int] = None) -> BPETokenizer:
    # XXX: base on number of processors available
    n_workers = n_workers or 4
    pool = Pool(n_workers)
    thread_pool = ThreadPool(n_workers)

    special_token_values: Dict[str, bytes] = { s : bytes(s) for s in special_tokens }
    
    vocab: List[bytes] = [bytes([b]) for b in range(256)]
    token_lookup: Dict[bytes, int] = { token: i for i, token in enumerate(vocab) }
    merges: List[Tuple[bytes, bytes]] = []
    doc_tokens: List[List[int]] = []

    token_presplits = pool.map(presplit_docs, _chunkify(docs, 10)) # XXX tune chunk size?
    for token_split in token_presplits:
        for doc in token_split.tokens:
            single_doc_tokens: List[int] = []
            for batch_token_id in doc:
                token_value: bytes = token_split.id_to_token[batch_token_id]
                if token_value not in token_lookup:
                    token_lookup[token_value] = len(vocab)
                    vocab.append(token_value)

                global_token_id: int = token_lookup[token_value]
                single_doc_tokens.append(global_token_id)

            doc_tokens.append(single_doc_tokens)

    byte_pair_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    top_counts: SortedSet[Tuple[int, Tuple[int, int]]] = SortedSet()

    # initialize all pairs (don't cross doc boundaries)
    for doc in doc_tokens:
        for i in range(len(doc) - 1):
            byte_pair_counts[tuple(doc[i:i+2])] += 1

    for pair, top_count in byte_pair_counts.items():
        top_counts.add((top_count, pair))

    while len(vocab) < vocab_size - len(special_tokens):
        if len(top_counts) == 0:
            print(f"exhausted all token pairs with vocab size of only {len(vocab)}")
            break
        
        top_count, top_pair = next(reversed(top_counts))
        t1, t2 = top_pair

        # print(f"removing {top_count} instances of pair {top_pair}")

        # register new token
        merges.append((vocab[t1], vocab[t2]))
        new_token: bytes = vocab[t1] + vocab[t2]
        token_lookup[new_token] = len(vocab)
        vocab.append(new_token)

        # print(f"created new token {new_token} with id {token_lookup[new_token]}")

        # merge token pair in each document and update pair counts accordingly
        # note: this modifies `doc` in place for efficiency
        merge_results = thread_pool.map(lambda docs: merge_tokens(docs, top_pair, token_lookup[new_token]), _chunkify(doc_tokens, 10))
        for pair_deltas in merge_results:
            for changed_pair, delta in pair_deltas.items():
                #print(f"changing count of {changed_pair} by {delta}")
                if changed_pair in byte_pair_counts:
                    top_counts.remove((byte_pair_counts[changed_pair], changed_pair))
                byte_pair_counts[changed_pair] += delta
                if byte_pair_counts[changed_pair] > 0:
                    top_counts.add((byte_pair_counts[changed_pair], changed_pair))

        # we should have removed all entries naturally as part of processing above
        for count, pair in top_counts:
            if pair == top_pair:
                print(f"still {count} pairs remaining of processed pair {top_pair}")
                assert False

    # XXX: add special tokens to end of vocab

    return BPETokenizer(vocab=vocab, merges=merges)

