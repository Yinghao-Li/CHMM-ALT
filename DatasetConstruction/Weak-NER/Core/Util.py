import spacy
import re
from Core.Constants import *

"""This module contains data structures useful for other modules, in particular tries (for searching for occurrences
of large terminology lists in texts) and interval trees (for representing annotations of text spans)."""


class Trie:
    """Implementation of a trie for searching for occurrences of terms in a text."""

    def __init__(self):
        self.start = Node()
        self.value_mapping = {}  # to avoid storing many (identical) strings as values, we use a mapping
        self.index_mapping = {}  # (reverse dictionary of value_mapping)
        self._length = 0

    def longest_prefix(self, key, case_sensitive=True):
        """Search for the longest prefix. The key must be a list of tokens. The method
        returns the prefix length (in number of covered tokens) and the corresponding value. """
        current = self.start
        value = None
        prefix_length = 0
        for i, c in enumerate(key):
            if current.children is None:
                break
            elif c in current.children:
                current = current.children[c]
                if current.value:
                    value = current.value
                    prefix_length = i + 1
            elif not case_sensitive:
                found_alternative = False
                for alternative in {c.title(), c.lower(), c.upper()}:
                    if alternative in current.children:
                        current = current.children[alternative]
                        if current.value:
                            value = current.value
                            prefix_length = i + 1
                        found_alternative = True
                        break
                if not found_alternative:
                    break
            else:
                break
        value = self.index_mapping[value] if value is not None else None
        return prefix_length, value

    def __contains__(self, key):
        return self[key] is not None

    def __getitem__(self, key):
        current = self.start
        for i, c in enumerate(key):
            if current.children is not None and c in current.children:
                current = current.children[c]
            else:
                return None
        return self.index_mapping.get(current.value, None)

    def __setitem__(self, key, value):
        current = self.start
        for c in key:
            if current.children is None:
                new_node = Node()
                current.children = {c: new_node}
                current = new_node
            elif c not in current.children:
                new_node = Node()
                current.children[c] = new_node
                current = new_node
            else:
                current = current.children[c]
        if value in self.value_mapping:
            value_index = self.value_mapping[value]
        else:
            value_index = len(self.value_mapping) + 1
            self.value_mapping[value] = value_index
            self.index_mapping[value_index] = value
        current.value = value_index
        self._length += 1

    def __len__(self):
        return self._length

    def __iter__(self):
        return self._iter_from_node(self.start)

    def _iter_from_node(self, n):
        if n.value is not None:
            yield (), n.value
        if n.children is not None:
            for child_key, child_value in n.children.items():
                for subval_key, subval_value in self._iter_from_node(child_value):
                    yield (child_key, *subval_key), subval_value

    def __repr__(self):
        return list(self).__repr__()


class Node:
    """Representation of a trie node"""
    __slots__ = ('children', 'value')

    def __init__(self):
        self.children = None
        self.value = None


def tokenise_fast(text):
    """Fast tokenisation of a string (designed to be roughly similar to Spacy's)"""
    ori_tokens = text.split(" ")

    regu_tokens = []
    for token in ori_tokens:

        # Special case: handle hyphenised tokens like Jean-Pierre
        if "-" in token:
            subtokens = token.split("-")
            for j, sub_token in enumerate(subtokens):
                regu_tokens.append(sub_token)
                if j < len(subtokens) - 1:
                    regu_tokens.append("-")

        # Special case: handle tokens like 3G, where Spacy tokenisation is unpredictable
        elif re.match(r"\d+[A-Za-z]+", token):
            if not hasattr(tokenise_fast, "nlp"):
                tokenise_fast.nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
            for tok in tokenise_fast.nlp(token):
                regu_tokens.append(tok.text)
        else:
            regu_tokens.append(token)

    i = 0
    while i < len(regu_tokens):

        # Special case: handle genitives
        if regu_tokens[i].endswith("'s"):
            regu_tokens[i] = regu_tokens[i].rstrip("s").rstrip("'")
            regu_tokens.insert(i + 1, "'s")
            i += 2
        else:
            i += 1

    regu_tokens = [token for token in regu_tokens if len(token) > 0]
    return regu_tokens


def is_likely_proper(tok):
    """Returns true if the spacy token is a likely proper name, based on its form."""
    if len(tok) < 2:
        return False

    # If the lemma is titled, just return True
    elif tok.lemma_.istitle():
        return True

    # Handling cases such as iPad
    elif len(tok) > 2 and tok.text[0].islower() and tok.text[1].isupper() and tok.text[2:].islower():
        return True

    elif (tok.is_upper and tok.text not in CURRENCY_CODES
          and tok.text not in NOT_NAMED_ENTITIES):
        return True

    # Else, check whether the surface token is titled and is not sentence-initial
    elif (tok.i > 0 and tok.is_title and not tok.is_sent_start and tok.nbor(-1).text not in {'\'', '"', '‘', '“', '”',
                                                                                             '’'}
          and not tok.nbor(-1).text.endswith(".")):
        return True
    return False


def in_compound(tok):
    """Returns true if the spacy token is part of a compound phrase"""
    if tok.dep_ == "compound":
        return True
    elif tok.i > 0 and tok.nbor(-1).dep_ == "compound":
        return True
    return False


def is_infrequent(span):
    """Returns true if there is at least one token with a rank > 15000"""
    max_rank = max([tok2.rank if tok2.rank > 0 else 20001 for tok2 in span])
    return max_rank > 15000


def get_spans(doc, sources, skip_overlaps=True):
    spans = set()
    for source in sources:
        if source not in doc.user_data["annotations"]:
            raise RuntimeError("Must run " + source + " first")
        for (start, end) in doc.user_data["annotations"][source]:
            spans.add((start, end))

    # If two spans are overlapping, return the longest spans
    finished = False
    while skip_overlaps and not finished:
        finished = True
        sorted_spans = sorted(spans, key=lambda x: x[0])
        for (start1, end1), (start2, end2) in zip(sorted_spans[:-1], sorted_spans[1:]):
            if start2 < end1:
                if (end1 - start1) > (end2 - start2):
                    spans.remove((start2, end2))
                else:
                    spans.remove((start1, end1))
                finished = False
                break
    return spans


def binary_search(start, end, intervals):
    """Performs a binary search"""
    start_search = 0
    end_search = len(intervals)
    while start_search < (end_search - 1):
        mid = start_search + (end_search - start_search) // 2
        (interval_start, interval_end) = intervals[mid]

        if interval_end <= start:
            start_search = mid
        elif interval_start >= end:
            end_search = mid
        else:
            break
    return start_search, end_search


def get_overlaps(start, end, annotations, sources=None):
    """Returns a list of overlaps (as (start, end, value) between the provided span
    and the existing annotations for the sources"""

    overlaps = []
    for source in (sources if sources is not None else annotations.keys()):
        intervals = list(annotations[source].keys())

        start_search, end_search = binary_search(start, end, intervals)

        for interval_start, interval_end in intervals[start_search:end_search]:
            if start < interval_end and end > interval_start:
                interval_value = annotations[source][(interval_start, interval_end)]
                overlaps.append((interval_start, interval_end, interval_value))

    return overlaps


def merge_contiguous_spans(spans, spacy_doc):
    """Merge spans that are contiguous (and with same label), or only separated with a comma"""

    finished = False
    while not finished:
        finished = True
        sorted_spans = sorted(spans, key=lambda x: x[0])
        for (start1, end1), (start2, end2) in zip(sorted_spans[:-1], sorted_spans[1:]):
            if end1 == start2 or (end1 == start2 - 1 and spacy_doc[end1].text == ","):
                val1 = spans[(start1, end1)]
                val2 = spans[start2, end2]
                if val1 == val2:
                    del spans[(start1, end1)]
                    del spans[(start2, end2)]
                    spans[(start1, end2)] = val1
                    finished = False
                    break
    return spans



