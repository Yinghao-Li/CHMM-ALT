import sys

import json
import re
import os
import logging
from typing import Optional
from xml.etree import ElementTree
from tqdm.auto import tqdm

from .util import format_text
from .data import formalize_bio


# noinspection PyUnboundLocalVariable
def load_conll_2003_data(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    docs = list()
    sent_list = None

    i = 0
    while i < len(lines):
        line = lines[i]
        try:
            token, _, _, ner_label = line.strip().split()
            if token == '-DOCSTART-':
                if sent_list:
                    docs.append({'sent_list': sent_list, 'labels_list': labels_list})
                sent_list = list()
                labels_list = list()
                sent = list()
                labels = list()
                i += 1
            else:
                sent.append(token)
                labels.append(ner_label)
        except ValueError:
            sent_list.append(sent)
            labels_list.append(labels)
            sent = list()
            labels = list()
        
        i += 1

    docs.append({'sent_list': sent_list, 'labels_list': labels_list})

    for doc in docs:
        sent_list = doc['sent_list']
        labels_list = doc['labels_list']
        for sentence, labels in zip(sent_list, labels_list):
            assert len(sentence) == len(labels)

    return docs


def load_wikigold_data(file_name):
    with open(file_name, 'r') as f:
        instances = json.load(f)

    sent_list = list()
    labels_list = list()
    for instant in instances:
        sent = instant['text']
        labels = formalize_bio(instant['labels'])
        sent_list.append(sent)
        labels_list.append(labels)

    for sentence, labels in zip(sent_list, labels_list):
        assert len(sentence) == len(labels)

    return sent_list, labels_list


def load_bc5cdr_sentences(file_name):
    root = ElementTree.parse(file_name).getroot()
    xml_docs = root.findall("./document")
    xml_sents = list()
    for xml_doc in tqdm(xml_docs):
        xml_title = xml_doc.find("passage[infon='title']")
        xml_abstract = xml_doc.find("passage[infon='abstract']")

        title = xml_title.find('text').text
        abstract = xml_abstract.find('text').text
        xml_sents.append(title + " " + abstract)
    return xml_sents


def load_ncbi_sentences(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    clusters = list()
    clines = None
    for line in lines:
        if line == '\n':
            if clines is not None:
                clusters.append(clines)
            clines = list()
        else:
            clines.append(line)

    sents = list()
    for src in clusters:
        src_txt = src[0].split('|')[2] + src[1].split('|')[2]
        sents.append(format_text(src_txt))

    return sents


# noinspection PyArgumentList
def set_logging(log_dir: Optional[str] = None):
    """
    setup logging
    Last modified: 07/20/21

    Parameters
    ----------
    log_dir: where to save logging file. Leave None to save no log files

    Returns
    -------

    """
    if log_dir is not None:
        if not os.path.isdir(os.path.split(log_dir)[0]):
            os.makedirs(os.path.abspath(os.path.normpath(os.path.split(log_dir)[0])))
        if os.path.isfile(log_dir):
            os.remove(log_dir)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir)
            ],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(sys.stdout)
            ],
        )


def prettify_json(text, indent=2, collapse_level=4):
    pattern = r"[\r\n]+ {%d,}" % (indent * collapse_level)
    text = re.sub(pattern, ' ', text)
    text = re.sub(r'([\[({])+ +', r'\g<1>', text)
    text = re.sub(r'[\r\n]+ {%d}([])}])' % (indent * (collapse_level-1)), r'\g<1>', text)
    text = re.sub(r'(\S) +([])}])', r'\g<1>\g<2>', text)
    return text
