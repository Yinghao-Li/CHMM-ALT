import re
import regex

import numpy as np

from typing import List, Optional
from nltk.tokenize import word_tokenize, sent_tokenize


def format_text(text):
    """
    Normalize text and transform some unicode characters into ascii

    Parameters
    ----------
    text: input text string

    Returns
    -------
    normalized text string
    """

    # deal with interpuncts
    interpunct = r'[\u00B7\u02D1\u0387\u05BC\u16EB\u2022\u2027\u2218\u2219\u22C5\u23FA' \
                 r'\u25CF\u25E6\u26AB\u2981\u2E30\u2E31\u2E33\u30FB\uA78F\uFF65]'
    text = re.sub(interpunct, ' ', text)

    # deal with bullets
    bullets = r'[\u2022\u2023\u2043\u204C\u204D\u2219\u25CB\u25D8\u25E6' \
              r'\u2619\u2765\u2767\u29BE\u29BF]'
    text = re.sub(bullets, ' ', text)

    # deal with overlay tilde
    tilde = r'[\u0303\u223C\u224B\u02DC\u02F7\u223D\u0360\u0334\u0330\u033E' \
            r'\u1DEC\uFE29\uFE2A\uFE22\uFE23]'
    text = re.sub(tilde, '~', text)

    # deal with overlay not tilde
    not_tilde = r'[\u034A]'
    text = re.sub(not_tilde, '≁', text)

    text = remove_combining_marks(text)

    # deal with invisible Soft Hyphen
    text = re.sub(r'[\u00ad]', ' ', text)

    # deal with white spaces and \n
    text = regex.sub(r'[\p{Z}]', ' ', text)
    text = re.sub(r'([ \t]+)?[\r\n]([ \t]+)?', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'([ ]{2,})', ' ', text)

    # deal with "/"
    text = re.sub(r'[ ]?/[ ]?', '/', text)

    # deal with dash/hyphen
    text = regex.sub(r'[\p{Pd}]', '-', text)
    text = re.sub(r'[\u2212\uf8ff\uf8fe\ue5f8]', '-', text)

    # convert user-defined characters to dash
    text = re.sub(r'[\uE000-\uF8FF]', '-', text)

    # deal with repeated comma and period
    text = re.sub(r'[.]+( *[,.])+', r'.', text)

    # deal with in-sentence references
    text = re.sub(r' *\[[0-9-, ]+][,-]*', r'', text)
    # deal with after-sentence references
    text = regex.sub(r'([A-Za-zα-ωΑ-Ω\p{Pe}\'"\u2018-\u201d]+[\d-]*)([.])( |[\d]+)([-, ]*[\d]*)*'
                     r'([ ]+[A-Z\dα-ωΑ-Ω\p{Ps}\'"\u2018-\u201d]|$)', r'\g<1>\g<2>\g<5>', text)

    text = text.strip()
    return text


def remove_combining_marks(text: str):
    """
    Remove combining marks (with unicode 0300-036f)

    Parameters
    ----------
    text: input string text

    Returns
    -------
    string text
    """
    # deal with interpuncts
    diacritics = r'[\u0300-\u036F]'
    text = re.sub(diacritics, '', text)

    return text


def substring_mapping(text: str, mapping_dict: dict):
    """
    Map substrings in the input string according to the dict

    Parameters
    ----------
    text: input string
    mapping_dict: the mapping dictionary

    Returns
    -------
    string text
    """
    rep = dict((re.escape(k), v) for k, v in mapping_dict.items())
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    return text


def break_overlength_bert_text(text: List[str], tokenizer, max_length: Optional[int] = 512):
    """
    Break the sentences that exceeds the maximum BERT length

    Parameters
    ----------
    text: A list of tokens that are in the original token format (instead of BERT BPE format)
    tokenizer: BERT tokenizer
    max_length: maximum BERT length

    Returns
    -------
    1.text_list: a list of broken text
    2. : the lengths of the broken text
    3. : a list of the indices of the broken text
    """

    # TODO: deal with overlength sentences (individual sentences longer than max_length)
    # Deal with sentences that are longer than 512 BERT tokens
    if len(tokenizer.tokenize(' '.join(text), add_special_tokens=True)) >= max_length:
        text_list = [text]
        bert_lengths = [len(tokenizer.tokenize(' '.join(t), add_special_tokens=True)) for t in text_list]
        while (np.asarray(bert_lengths) >= max_length).any():
            splitted_text_list = list()
            for tokens, bert_len in zip(text_list, bert_lengths):

                if bert_len < max_length:
                    splitted_text_list.append(tokens)
                    continue
                splitted_text = sent_tokenize(' '.join(tokens))

                sent_lens = list()
                for st in splitted_text:
                    sent_lens.append(len(word_tokenize(st)))
                ends = [np.sum(sent_lens[:i]) for i in range(1, len(sent_lens) + 1)]

                halfway_idx = np.argmin((np.array(ends) - len(tokens) / 2) ** 2)
                splitted_text_list.append(tokens[:ends[halfway_idx]])
                splitted_text_list.append(tokens[ends[halfway_idx]:])

            text_list = splitted_text_list
            bert_lengths = [len(tokenizer.tokenize(' '.join(t), add_special_tokens=True)) for t in text_list]

        text_lenghts = [len(txt) for txt in text_list]
        assert np.sum(text_lenghts) == len(text), ValueError(f'Text splitting failed: {text} ---> {text_list}')
        return text_list, text_lenghts, np.arange(len(text_list))
    else:
        return [text], [len(text)], np.array([0], dtype=np.int)
