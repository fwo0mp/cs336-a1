from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Set, Tuple

from sortedcontainers import SortedSet

@dataclass
class BPETokenizer:
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]


def _presplit_tokens(text: str) -> List[bytes]:
    pass


def _merge_tokens(doc: List[int], old_pair: Tuple[int, int], new_token: int) -> Dict[Tuple[int, int], int]:
    """
        !!! modifies doc in-place, replacing all instances of old_pair with new_token

        returns deltas of token pair counts to be consolidated by central coordinating thread

        factored into separate function to make more conducive to parallelizing, replacing with rust implementation, etc.
        (or even rpc call to distribute across cluster)
    """
    write_cursor: int = 0
    read_cursor: int = 0
    deltas: Dict[Tuple[int, int], int] = defaultdict(int)

    while read_cursor < len(doc):
        if read_cursor < len(doc) - 1 and tuple(doc[read_cursor:read_cursor + 2]) == old_pair:
            deltas[tuple(doc[read_cursor:read_cursor + 2])] -= 1

            # decrement counts of pairs that were previously formed on either side
            # and replace them with newly formed pairs from substituting the new token
            if read_cursor > 0:
                deltas[tuple(doc[read_cursor - 1:read_cursor + 1])] -= 1
                deltas[(doc[read_cursor - 1], new_token)] += 1
            if read_cursor < len(doc) - 2:
                deltas[tuple(doc[read_cursor + 1:read_cursor + 3])] -= 1
                deltas[(new_token, doc[read_cursor + 2])] += 1

            doc[write_cursor] = new_token
            read_cursor += 2
        else:
            doc[write_cursor] = doc[read_cursor]
            read_cursor += 1

        # we are writing one new token regardless
        write_cursor += 1
    
    # truncate document based on how many tokens we wrote
    doc[write_cursor:] = []
    
    return deltas
    

def train_tokenizer(docs: List[str], vocab_size: int, special_tokens: Set[str]) -> BPETokenizer:
    doc_tokens: List[List[bytes]] = map(_presplit_tokens, docs)
    special_token_values: Dict[str, bytes] = { s : bytes(s) for s in special_tokens }
    
    # leave space for inserting end of doc special tokens
    total_length: int = sum(map(len, doc_tokens)) + len(doc_tokens)

    initial_tokens: Set[bytes] = set()
    for doc in doc_tokens:
        initial_tokens.add(doc)
    vocab: List[bytes] = [bytes([b]) for b in range(256)] + list(initial_tokens)
    token_lookup: Dict[bytes, int] = { token: i for i, token in vocab }
    merges: List[Tuple[bytes, bytes]] = []

    byte_pair_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    top_counts: SortedSet[Tuple[int, Tuple[int, int]]] = SortedSet()

    # initialize all pairs (don't cross doc boundaries)
    for doc in doc_tokens:
        for i in range(len(doc) - 1):
            t1, t2 = doc[i:i+1]
            pair = (token_lookup[t1], token_lookup[t2])
            byte_pair_counts[pair] += 1

    for pair, count in byte_pair_counts.items():
        top_counts.add((count, pair))

    while len(vocab) < vocab_size - len(special_tokens):
        if len(top_counts) == 0:
            print(f"exhausted all token pairs with vocab size of only {len(vocab)}")
            break
        
        count, top_pair = next(reversed(top_counts))
        t1, t2 = top_pair
        del top_counts[(count, top_pair)]

        merges.append(top_pair)

        # register new token
        new_token: bytes = t1 + t2
        vocab.append(new_token)
        token_lookup[new_token] = len(vocab)

        # merge token pair in each document and update pair counts accordingly
        for doc in doc_tokens:
            # note: this modifies `doc` in place for efficiency
            pair_deltas = _merge_tokens(doc, top_pair, new_token)
            for changed_pair, delta in pair_deltas.items():
                if changed_pair in byte_pair_counts:
                    top_counts.remove((byte_pair_counts[changed_pair], changed_pair))
                byte_pair_counts[changed_pair] += delta
                top_counts.add((byte_pair_counts[changed_pair], changed_pair))

    # XXX: add special tokens to end of vocab

    return BPETokenizer(vocab=vocab, merges=merges)

