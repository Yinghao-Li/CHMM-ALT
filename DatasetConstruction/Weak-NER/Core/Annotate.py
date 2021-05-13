import itertools
import pickle
import re

import numpy as np
import snips_nlu_parsers
import spacy
import spacy.attrs
import spacy.tokens
from typing import List

from Core.IO import docbin_reader, extract_json_data
from Core.Util import Trie, is_likely_proper, get_overlaps, merge_contiguous_spans, \
    in_compound, get_spans, is_infrequent
from Core.SpacyWrapper import correct_entities
from Core.Constants import *


class BaseAnnotator:
    """Base class for the annotations.  """

    def __init__(self, to_exclude=None):
        self.to_exclude = list(to_exclude) if to_exclude is not None else []

    def pipe(self, docs):
        """Goes through the stream of documents and annotate them"""

        for doc in docs:
            yield self.annotate(doc)

    def annotate(self, doc):
        """Annotates one single document"""

        raise NotImplementedError()

    @staticmethod
    def clear_source(doc, source):
        """Clears the annotation associated with a given lb_source name"""

        if "annotations" not in doc.user_data:
            doc.user_data["annotations"] = {}
        doc.user_data["annotations"][source] = {}

    def add(self, doc, start, end, label, source, conf=1.0):
        """ Adds a labelled span to the annotation"""

        if not self._is_allowed_span(doc, start, end):
            return
        elif (start, end) not in doc.user_data["annotations"][source]:
            doc.user_data["annotations"][source][(start, end)] = ((label, conf),)

        # If the span is already present, we need to check that the total confidence does not exceed 1.0
        else:
            current_vals = doc.user_data["annotations"][source][(start, end)]
            if label in {label2 for label2, _ in current_vals}:
                return
            total_conf = sum([conf2 for _, conf2 in current_vals]) + conf
            if total_conf > 1.0:
                current_vals = [(label2, conf2 / total_conf) for label2, conf2 in current_vals]
                conf = conf / total_conf
            doc.user_data["annotations"][source][(start, end)] = (*current_vals, (label, conf))

    def _is_allowed_span(self, doc, start, end):
        """Checks whether the span is allowed (given exclusivity relations with other sources)"""
        for source in self.to_exclude:
            intervals = list(doc.user_data["annotations"][source].keys())

            start_search, end_search = self._binary_search(start, end, intervals)
            for interval_start, interval_end in intervals[start_search:end_search]:
                if start < interval_end and end > interval_start:
                    return False

        return True

    def annotate_docbin(self, docbin_input_file, docbin_output_file=None, return_raw=False,
                        cutoff=None, nb_to_skip=0):
        """Runs the annotator on the documents of a DocBin file, and write the output
        to the same file (or returns the raw data is return_raw is True)"""

        attrs = [spacy.attrs.LEMMA, spacy.attrs.TAG, spacy.attrs.DEP, spacy.attrs.HEAD,
                 spacy.attrs.ENT_IOB, spacy.attrs.ENT_TYPE]
        docbin = spacy.tokens.DocBin(attrs=attrs, store_user_data=True)

        print("Reading", docbin_input_file, end="...", flush=True)
        for doc in self.pipe(docbin_reader(docbin_input_file, cutoff=cutoff, nb_to_skip=nb_to_skip)):
            docbin.add(doc)
            if len(docbin) % 1000 == 0:
                print("Number of processed documents:", len(docbin))

        print("Finished annotating", docbin_input_file)

        data = docbin.to_bytes()
        if return_raw:
            return data
        else:
            if docbin_output_file is None:
                docbin_output_file = docbin_input_file
            print("Write to", docbin_output_file, end="...", flush=True)
            fd = open(docbin_output_file, "wb")
            fd.write(data)
            fd.close()
            print("done")

    @staticmethod
    def _binary_search(start, end, intervals):
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


class UnifiedAnnotator(BaseAnnotator):
    """
    Base class for annotators that seek to 'unify' the annotations from other
    supervision sources. Must implement two class: 'train' and 'label'
    """

    def __init__(self, sources_to_keep=None, source_name="HMM"):
        BaseAnnotator.__init__(self)
        if sources_to_keep is None:
            self.source_indices_to_keep = {i for i in range(len(SOURCE_NAMES))}
        else:
            print("Using", sources_to_keep, "as supervision sources")
            self.source_indices_to_keep = {SOURCE_NAMES.index(s) for s in sources_to_keep}
        self.source_name = source_name

    def train(self, docbin_file, cutoff=None):
        """
        Trains the parameters of the annotator based on the annotation in the provided
        docbin file
        """

        raise NotImplementedError()

    def label(self, doc):
        """
        Returns two lists, one list with the predicted label for each token in the document,
        and one with the associated probabilities according to the model.
        """

        raise NotImplementedError()

    def annotate(self, doc):
        """
        Annotates the document with a new layer of annotation with the model predictions
        """

        doc.user_data["annotations"][self.source_name] = {}

        predicted, confidences = self.label(doc)
        i = 0
        while i < len(predicted):
            if predicted[i] != "O":
                if predicted[i].startswith("U-") or predicted[i].startswith("I-") or predicted[i].startswith("L-"):
                    conf = round(confidences[i], 3)
                    doc.user_data["annotations"][self.source_name][(i, i + 1)] = ((predicted[i][2:], conf),)
                    i += 1
                elif predicted[i].startswith("B-"):
                    start = i
                    label = predicted[i][2:]
                    i += 1
                    while i < (len(doc) - 1) and predicted[i] != "O" and predicted[i].startswith("I-"):
                        i += 1
                    if i < len(doc) and predicted[i].startswith("L-"):
                        conf = round(confidences[start:i + 1].max().mean(), 3)
                        doc.user_data["annotations"][self.source_name][(start, i + 1)] = ((label, conf),)
                    i += 1
            else:
                i += 1
        return doc

    def extract_sequence(self, doc):
        """
        Convert the annotations of a spacy document into an array of observations of shape
        (nb_sources, nb_biluo_labels)
        """

        doc = self.specialise_annotations(doc)
        sequence = np.zeros((len(doc), len(SOURCE_NAMES), len(POSITIONED_LABELS_BIO)), dtype=np.float32)
        for i, source in enumerate(SOURCE_NAMES):
            sequence[:, i, 0] = 1.0
            if source not in doc.user_data["annotations"] or i not in self.source_indices_to_keep:
                continue
            for (start, end), vals in doc.user_data["annotations"][source].items():
                for label, conf in vals:
                    # Such condition should not exist
                    if label in {"MISC", "ENT"}:
                        continue
                    elif start >= len(doc):
                        print("wrong boundary")
                        continue
                    elif end > len(doc):
                        print("wrong boundary2")
                        end = len(doc)
                    sequence[start:end, i, 0] = 0.0
                    sequence[start, i, LABEL_INDICES["B-%s" % label]] = conf
                    if end - start > 1:
                        sequence[start + 1: end, i, LABEL_INDICES["I-%s" % label]] = conf

        return sequence

    def specialise_annotations(self, doc):
        """
        Replace generic ENT or MISC values with the most likely labels from other annotators
        """

        to_add = []

        annotated = doc.user_data.get("annotations", [])
        for source in annotated:

            other_sources = [s for s in annotated if "HMM" not in s
                             and s != "gold" and s in SOURCE_NAMES
                             and s != source and "proper" not in s and "nnp_" not in s
                             and "SEC" not in s
                             and "compound" not in s and "BTC" not in s
                             and SOURCE_NAMES.index(s) in self.source_indices_to_keep]

            current_spans = dict(annotated[source])
            for (start, end), vals in current_spans.items():
                for label, conf in vals:
                    if label in {"ENT", "MISC"}:

                        label_counts = {}
                        for other_source in other_sources:
                            overlaps = get_overlaps(start, end, annotated, [other_source])
                            for (start2, end2, vals2) in overlaps:
                                for label2, conf2 in vals2:
                                    if label2 not in {"ENT", "MISC"}:
                                        # simple heuristic for the confidence of the label specialisation
                                        conf2 = conf2 if (start2 == start and end2 == end) else 0.3 * conf2
                                        conf2 = conf2 * SOURCE_PRIORS[other_source][label2][0]
                                        label_counts[label2] = label_counts.get(label2, 0) + conf * conf2
                        vals = tuple((lb, SOURCE_PRIORS[source][lb][0] * conf2 / sum(label_counts.values()))
                                     for lb, conf2 in label_counts.items())
                        to_add.append((source, start, end, vals))

        for source, start, end, vals in to_add:
            doc.user_data["annotations"][source][(start, end)] = vals

        return doc

    def save(self, filename):
        fd = open(filename, "wb")
        pickle.dump(self, fd)
        fd.close()

    @classmethod
    def load(cls, pickle_file):
        print("Loading", pickle_file)
        fd = open(pickle_file, "rb")
        ua = pickle.load(fd)
        fd.close()
        return ua


class UnitedAnnotator(BaseAnnotator):
    """Annotator of entities in documents, combining several sub-annotators (such as gazetteers,
    spacy models etc.). To add all annotators currently implemented, call add_all(). """

    def __init__(self):
        super(UnitedAnnotator, self).__init__()
        self.annotators = []

    def annotate(self, doc):
        """Annotates a single  document with the sub-annotators
        NB: do not use this method for large collections of documents (as it is quite inefficient), and
        prefer the method pipe that runs the Spacy models on batches of documents"""

        for annotator in self.annotators:
            doc = annotator.annotate(doc)
        return doc

    def add_annotator(self, annotator):
        self.annotators.append(annotator)
        return self

    def add_all(self):
        """Adds all implemented annotation functions, models and filters"""
        print("[INFO] Start loading labeling models...")
        print("Loading weak lebelling functions")
        self.add_labeling_func()
        print("Loading Spacy NER models")
        self.add_models()
        print("Loading gazetteer supervision modules")
        self.add_gazetteers()
        print("Loading document-level labeling sources")
        self.add_doc_level()
        print("[INFO] All models are successfully loaded")

        return self

    def add_labeling_func(self):
        """Adds shallow annotation functions"""

        # Detection of dates, time, money, and numbers
        self.add_annotator(FunctionAnnotator(date_generator, "date_detector"))
        self.add_annotator(FunctionAnnotator(time_generator, "time_detector"))
        self.add_annotator(FunctionAnnotator(money_generator, "money_detector"))
        exclusives = ["date_detector", "time_detector", "money_detector"]

        # Detection based on casing
        proper_detector = SpanGenerator(is_likely_proper)
        self.add_annotator(FunctionAnnotator(proper_detector, "proper_detector",
                                             to_exclude=exclusives))

        # Detection based on casing, but allowing some lowercased tokens
        proper2_detector = SpanGenerator(is_likely_proper, exceptions=LOWERCASED_TOKENS)
        self.add_annotator(FunctionAnnotator(proper2_detector, "proper2_detector",
                                             to_exclude=exclusives))

        # Detection based on part-of-speech tags
        nnp_detector = SpanGenerator(lambda tok: tok.tag_ == "NNP")
        self.add_annotator(FunctionAnnotator(nnp_detector, "nnp_detector",
                                             to_exclude=exclusives))

        # Detection based on dependency relations (compound phrases)
        compound_detector = SpanGenerator(lambda x: is_likely_proper(x) and in_compound(x))
        self.add_annotator(FunctionAnnotator(compound_detector, "compound_detector",
                                             to_exclude=exclusives))

        # We add one variants for each NE detector, looking at infrequent tokens
        for source_name in ["proper_detector", "proper2_detector", "nnp_detector", "compound_detector"]:
            self.add_annotator(SpanConstraintAnnotator(is_infrequent, source_name, "infrequent_"))

        self.add_annotator(FunctionAnnotator(legal_generator, "legal_detector", exclusives))
        exclusives += ["legal_detector"]
        self.add_annotator(FunctionAnnotator(number_generator, "number_detector", exclusives))

        # Detection of companies with a legal type
        self.add_annotator(FunctionAnnotator(CompanyTypeGenerator(), "company_type_detector",
                                             to_exclude=exclusives))

        # Detection of full person names
        self.add_annotator(FunctionAnnotator(FullNameGenerator(), "full_name_detector",
                                             to_exclude=exclusives + ["company_type_detector"]))

        # Detection based on a probabilistic parser
        self.add_annotator(FunctionAnnotator(SnipsGenerator(), "snips"))

        return self

    def add_models(self):
        """Adds Spacy NER models to the annotator"""

        self.add_annotator(ModelAnnotator("en_core_web_md", "core_web_md"))
        self.add_annotator(ModelAnnotator("data/conll2003", "conll2003"))
        self.add_annotator(ModelAnnotator("data/BTC", "BTC"))
        self.add_annotator(ModelAnnotator("data/SEC-filings", "SEC"))

        return self

    def add_gazetteers(self):
        """Adds gazetteer supervision models (company names and wikidata)."""

        exclusives = ["date_detector", "time_detector", "money_detector", "number_detector"]

        # Annotation of company, person and location names based on wikidata
        self.add_annotator(GazetteerAnnotator(WIKIDATA, "wiki", to_exclude=exclusives))

        # Annotation of company, person and location names based on wikidata (only entries with descriptions)
        self.add_annotator(GazetteerAnnotator(WIKIDATA_SMALL, "wiki_small", to_exclude=exclusives))

        # Annotation of location names based on geonames
        self.add_annotator(GazetteerAnnotator(GEONAMES, "geo", to_exclude=exclusives))

        # Annotation of organisation and person names based on crunchbase open data
        self.add_annotator(GazetteerAnnotator(CRUNCHBASE, "crunchbase", to_exclude=exclusives))

        # Annotation of product names
        self.add_annotator(GazetteerAnnotator(PRODUCTS, "product", to_exclude=exclusives[:-1]))

        # We also add new sources for multitoken entities (which have higher confidence)
        for source_name in ["wiki", "wiki_small", "geo", "crunchbase", "product"]:
            for cased in ["cased", "uncased"]:
                self.add_annotator(
                    SpanConstraintAnnotator(lambda s: len(s) > 1, "%s_%s" % (source_name, cased), "multitoken_"))

        self.add_annotator(FunctionAnnotator(misc_generator, "misc_detector", exclusives))

        return self

    def add_doc_level(self):
        """Adds document-level supervision sources"""

        self.add_annotator(StandardiseAnnotator())
        self.add_annotator(DocumentHistoryAnnotator())
        self.add_annotator(DocumentMajorityAnnotator())
        return self


# ---------------------------------------------------------------- #
# Annotators trained with labeled data
class ModelAnnotator(BaseAnnotator):
    """Annotation based on a spacy NER model"""

    def __init__(self, model_path, source_name):
        super(ModelAnnotator, self).__init__()
        print("loading", model_path, end="...", flush=True)
        model = spacy.load(model_path)
        self.ner = model.get_pipe("ner")
        self.source_name = source_name
        print("done")

    def pipe(self, docs):
        """Annotates the stream of documents based on the Spacy NER model"""

        stream1, stream2 = itertools.tee(docs, 2)

        # Apply the NER models through the pipe
        # (we need to work on copies to strange deadlock conditions)
        stream2 = (spacy.tokens.Doc(d.vocab).from_bytes(d.to_bytes(exclude="user_data"))
                   for d in stream2)

        def remove_ents(doc_):
            doc_.ents = tuple()
            return doc_

        stream2 = (remove_ents(d) for d in stream2)
        stream2 = self.ner.pipe(stream2)

        for doc, doc_copy in zip(stream1, stream2):

            self.clear_source(doc, self.source_name)
            self.clear_source(doc, self.source_name + "+c")

            # Add the annotation
            for ent in doc_copy.ents:
                self.add(doc, ent.start, ent.end, ent.label_, self.source_name)

            # Correct some entities
            doc_copy = correct_entities(doc_copy)
            for ent in doc_copy.ents:
                self.add(doc, ent.start, ent.end, ent.label_, self.source_name + "+c")

            yield doc

    def annotate(self, doc):
        """Annotates one single document using the Spacy NER model
        NB: do not use this method for large collections of documents (as it is quite inefficient), and
        prefer the method pipe that runs the Spacy model on batches of documents"""

        ents = list(doc.ents)
        doc.ents = tuple()
        doc = self.ner(doc)

        self.clear_source(doc, self.source_name)
        self.clear_source(doc, self.source_name + "+c")

        # Add the annotation
        for ent in doc.ents:
            self.add(doc, ent.start, ent.end, ent.label_, self.source_name)

        # Correct some entities
        doc = correct_entities(doc)
        for ent in doc.ents:
            self.add(doc, ent.start, ent.end, ent.label_, self.source_name + "+c")

        doc.ents = ents
        return doc


# ---------------------------------------------------------------- #
# Individual weak annotators
class GazetteerAnnotator(BaseAnnotator):
    """Annotation using a gazetteer, i.e. a large list of entity terms. The annotation looks
    both at case-sensitive and case-insensitive occurrences.  The annotator relies on a token-level
    trie for efficient search. """

    def __init__(self, json_file, source_name, to_exclude=None):

        super(GazetteerAnnotator, self).__init__(to_exclude=to_exclude)

        self.trie = extract_json_data(json_file)
        self.source_name = source_name

    def annotate(self, doc):
        """Annotates one single document"""

        self.clear_source(doc, "%s_%s" % (self.source_name, "cased"))
        self.clear_source(doc, "%s_%s" % (self.source_name, "uncased"))

        for start, end, label, conf in self.get_hits(doc, case_sensitive=True, full_compound=True):
            self.add(doc, start, end, label, "%s_%s" % (self.source_name, "cased"), conf)

        for start, end, label, conf in self.get_hits(doc, case_sensitive=False, full_compound=True):
            self.add(doc, start, end, label, "%s_%s" % (self.source_name, "uncased"), conf)

        return doc

    def get_hits(self, spacy_doc, case_sensitive=True, lookahead=10, full_compound=True):
        """Search for occurrences of entity terms in the spacy document"""

        tokens = tuple(tok.text for tok in spacy_doc)

        i = 0
        while i < len(tokens):

            tok = spacy_doc[i]
            # Skip punctuation
            if tok.is_punct:
                i += 1
                continue

            # We skip if we are inside a compound phrase
            elif full_compound and i > 0 and is_likely_proper(spacy_doc[i - 1]) and spacy_doc[i - 1].dep_ == "compound":
                i += 1
                continue

            span = tokens[i:i + lookahead]
            prefix_length, prefix_value = self.trie.longest_prefix(span, case_sensitive)
            if prefix_length:

                # We further require at least one proper noun token (to avoid too many FPs)
                if not any(is_likely_proper(tok) for tok in spacy_doc[i:i + prefix_length]):
                    i += 1
                    continue

                # If we found a company and the next token is a legal suffix, include it
                if ((i + prefix_length) < len(spacy_doc) and {"ORG", "COMPANY"}.intersection(prefix_value)
                        and spacy_doc[i + prefix_length].lower_.rstrip(".") in LEGAL_SUFFIXES):
                    prefix_length += 1

                # If the following tokens are part of the same compound phrase, skip
                if full_compound and spacy_doc[i + prefix_length - 1].dep_ == "compound" and \
                        spacy_doc[i + prefix_length].text not in {"'s"}:
                    i += 1
                    continue

                # Must account for spans with multiple possible entities
                for neClass in prefix_value:
                    yield i, i + prefix_length, neClass, 1 / len(prefix_value)

                # We skip the text until the end of the occurences + 1 token (we assume two entities do not
                # follow one another without at least one token such as a comma)
                i += (prefix_length + 1)
            else:
                i += 1


# ---------------------------------------------------------------- #
# Labeling function annotator and pre-defined labeling functions
class FunctionAnnotator(BaseAnnotator):
    """Annotation based on a heuristic function that generates (start,end,label) given a spacy document"""

    def __init__(self, function, source_name, to_exclude=None):
        """Create an annotator based on a function generating labelled spans given a Spacy Doc object. Spans that
        overlap with existing spans from sources listed in 'to_exclude' are ignored. """

        super(FunctionAnnotator, self).__init__(to_exclude=to_exclude)

        self.function = function
        self.source_name = source_name

    def annotate(self, doc):
        """Annotates one single document"""

        self.clear_source(doc, self.source_name)

        for start, end, label in self.function(doc):
            self.add(doc, start, end, label, self.source_name)
        return doc


def date_generator(spacy_doc):
    """Searches for occurrences of date patterns in text"""

    spans = {}

    i = 0
    while i < len(spacy_doc):
        tok = spacy_doc[i]
        if tok.lemma_ in DAYS | DAYS_ABBRV:
            spans[(i, i + 1)] = "DATE"
        elif tok.is_digit and re.match(r"\d+$", tok.text) and 1920 < int(tok.text) < 2040:
            spans[(i, i + 1)] = "DATE"
        elif tok.lemma_ in MONTHS | MONTHS_ABBRV:
            if tok.tag_ == "MD":  # Skipping "May" used as auxiliary
                pass
            elif i > 0 and \
                    re.match(r"\d+$", spacy_doc[i - 1].text) and \
                    int(spacy_doc[i - 1].text) < 32:
                spans[(i - 1, i + 1)] = "DATE"
            elif i > 1 and \
                    re.match(r"\d+(?:st|nd|rd|th)$", spacy_doc[i - 2].text) and \
                    spacy_doc[i - 1].lower_ == "of":
                spans[(i - 2, i + 1)] = "DATE"
            elif i < len(spacy_doc) - 1 and \
                    re.match(r"\d+$", spacy_doc[i + 1].text) and \
                    int(spacy_doc[i + 1].text) < 32:
                spans[(i, i + 2)] = "DATE"
                i += 1
            else:
                spans[(i, i + 1)] = "DATE"
        i += 1

    # Concatenating contiguous spans
    spans = merge_contiguous_spans(spans, spacy_doc)

    for i, ((start, end), content) in enumerate(spans.items()):
        yield start, end, content


def time_generator(spacy_doc):
    """Searches for occurrences of time patterns in text"""

    i = 0
    while i < len(spacy_doc):
        tok = spacy_doc[i]

        if i < len(spacy_doc) - 1 and \
                tok.text[0].isdigit() and \
                spacy_doc[i + 1].lower_ in {"am", "pm", "a.m.", "p.m.", "am.", "pm."}:
            yield i, i + 2, "TIME"
            i += 1
        elif tok.text[0].isdigit() and re.match(r"\d{1,2}:\d{1,2}", tok.text):
            yield i, i + 1, "TIME"
            i += 1
        i += 1


def money_generator(spacy_doc):
    """Searches for occurrences of money patterns in text"""

    i = 0
    while i < len(spacy_doc):
        tok = spacy_doc[i]
        if tok.text[0].isdigit():
            j = i + 1
            while j < len(spacy_doc) and (spacy_doc[j].text[0].isdigit() or spacy_doc[j].norm_ in MAGNITUDES):
                j += 1

            found_symbol = False
            if i > 0 and spacy_doc[i - 1].text in (CURRENCY_CODES | CURRENCY_SYMBOLS):
                i = i - 1
                found_symbol = True
            if j < len(spacy_doc) and spacy_doc[j].text in (CURRENCY_CODES | CURRENCY_SYMBOLS |
                                                            {"euros", "cents", "rubles"}):
                j += 1
                found_symbol = True

            if found_symbol:
                yield i, j, "MONEY"
            i = j
        else:
            i += 1


def number_generator(spacy_doc):
    """Searches for occurrences of number patterns (cardinal, ordinal, quantity or percent) in text"""

    i = 0
    while i < len(spacy_doc):
        tok = spacy_doc[i]

        if tok.lower_ in ORDINALS:
            yield i, i + 1, "ORDINAL"

        elif re.search(r"\d", tok.text):
            j = i + 1
            while j < len(spacy_doc) and (spacy_doc[j].norm_ in MAGNITUDES):
                j += 1
            if j < len(spacy_doc) and spacy_doc[j].lower_.rstrip(".") in UNITS:
                j += 1
                yield i, j, "QUANTITY"
            elif j < len(spacy_doc) and spacy_doc[j].lower_ in ["%", "percent", "pc.", "pc", "pct", "pct.", "percents",
                                                                "percentage"]:
                j += 1
                yield i, j, "PERCENT"
            else:
                yield i, j, "CARDINAL"
            i = j - 1
        i += 1


class SpanGenerator:
    """Generate spans that satisfy a token-level constratint"""

    def __init__(self, constraint, label="ENT", exceptions=("'s", "-")):
        """annotation with a constraint (on spacy tokens). Exceptions are sets of tokens that are allowed
        to violate the constraint inside the span"""

        self.constraint = constraint
        self.label = label
        self.exceptions = set(exceptions)

    def __call__(self, spacy_doc):

        i = 0
        while i < len(spacy_doc):
            tok = spacy_doc[i]
            # We search for the longest span that satisfy the constraint
            if self.constraint(tok):
                j = i + 1
                while True:
                    if j < len(spacy_doc) and self.constraint(spacy_doc[j]):
                        j += 1
                    # We relax the constraint a bit to allow genitive and dashes
                    elif j < (len(spacy_doc) - 1) and spacy_doc[j].text in self.exceptions and self.constraint(
                            spacy_doc[j + 1]):
                        j += 2
                    else:
                        break

                # To avoid too many FPs, we only keep entities with at least 3 characters (excluding punctuation)
                if len(spacy_doc[i:j].text.rstrip(".")) > 2:
                    yield i, j, self.label
                i = j
            else:
                i += 1


class CompanyTypeGenerator:
    """Search for compound spans that end with a legal suffix"""

    def __init__(self):
        self.suggest_generator = SpanGenerator(lambda x: is_likely_proper(x) and in_compound(x))

    def __call__(self, spacy_doc):

        for start, end, _ in self.suggest_generator(spacy_doc):
            if spacy_doc[end - 1].lower_.rstrip(".") in LEGAL_SUFFIXES:
                yield start, end, "COMPANY"
            elif end < len(spacy_doc) and spacy_doc[end].lower_.rstrip(".") in LEGAL_SUFFIXES:
                yield start, end + 1, "COMPANY"


class FullNameGenerator:
    """Search for occurrences of full person names (first name followed by at least one title token)"""

    def __init__(self):
        fd = open(FIRST_NAMES)
        self.first_names = set(json.load(fd))
        fd.close()
        self.suggest_generator = SpanGenerator(lambda x: is_likely_proper(x) and in_compound(x),
                                               exceptions=NAME_PREFIXES)

    def __call__(self, spacy_doc):

        for start, end, _ in self.suggest_generator(spacy_doc):

            # We assume full names are between 2 and 4 tokens
            if (end - start) < 2 or (end - start) > 5:
                continue

            elif (spacy_doc[start].text in self.first_names and spacy_doc[end - 1].is_alpha
                  and spacy_doc[end - 1].is_title):
                yield start, end, "PERSON"


class SnipsGenerator:
    """Annotation using the Snips NLU entity parser. """

    def __init__(self):
        """Initialise the annotation tool."""

        self.parser = snips_nlu_parsers.BuiltinEntityParser.build(language="en")

    def __call__(self, spacy_doc):
        """Runs the parser on the spacy document, and convert the result to labels."""

        text = spacy_doc.text

        # The current version of Snips has a bug that makes it crash with some rare Turkish characters, or mentions
        # of "billion years"
        text = text.replace("’", "'").replace("”", "\"").replace("“", "\"").replace("—", "-").encode(
            "iso-8859-15", "ignore").decode("iso-8859-15")
        text = re.sub(r"(\d+) ([bm]illion(?: (?:\d+|one|two|three|four|five|six|seven|eight|nine|ten))? years?)",
                      r"\g<1>.0 \g<2>", text)

        results = self.parser.parse(text)
        for result in results:
            span = spacy_doc.char_span(result["range"]["start"], result["range"]["end"])
            if span is None or span.lower_ in {"now"} or span.text in {"may"}:
                continue
            label = None
            if result["entity_kind"] == "snips/number" and span.lower_ not in {"one", "some", "few", "many", "several"}:
                label = "CARDINAL"
            elif result["entity_kind"] == "snips/ordinal" and span.lower_ not in {"first", "second", "the first",
                                                                                  "the second"}:
                label = "ORDINAL"
            elif result["entity_kind"] == "snips/amountOfMoney":
                label = "MONEY"
            elif result["entity_kind"] == "snips/percentage":
                label = "PERCENT"
            elif result["entity_kind"] in {"snips/date", "snips/datePeriod"}:
                label = "DATE"
            elif result["entity_kind"] in {"snips/time", "snips/timePeriod"}:
                label = "TIME"

            if label:
                yield span.start, span.end, label


def legal_generator(spacy_doc):
    legal_spans = {}
    for (start, end) in get_spans(spacy_doc, ["proper2_detector", "nnp_detector"]):
        if not is_likely_proper(spacy_doc[end - 1]):
            continue

        last_token = spacy_doc[end - 1].text.title().rstrip("s")

        if last_token in LEGAL:
            legal_spans[(start, end)] = "LAW"

    # Handling legal references such as Article 5
    for i in range(len(spacy_doc) - 1):
        if spacy_doc[i].text.rstrip("s") in {"Article", "Paragraph", "Section", "Chapter", "§"}:
            if spacy_doc[i + 1].text[0].isdigit() or spacy_doc[i + 1].text in ROMAN_NUMERALS:
                start, end = i, i + 2
                if (i < len(spacy_doc) - 3 and spacy_doc[i + 2].text in {"-", "to", "and"}
                        and (spacy_doc[i + 3].text[0].isdigit() or spacy_doc[i + 3].text in ROMAN_NUMERALS)):
                    end = i + 4
                legal_spans[start, end] = "LAW"

    # Merge contiguous spans of legal references ("Article 5, Paragraph 3")
    legal_spans = merge_contiguous_spans(legal_spans, spacy_doc)
    for start, end in legal_spans:
        yield start, end, "LAW"


def misc_generator(spacy_doc):
    """Detects occurrences of countries and various less-common entities (NORP, FAC, EVENT, LANG)"""

    spans = set(spacy_doc.user_data["annotations"]["proper_detector"].keys())
    spans.update((i, i + 1) for i in range(len(spacy_doc)))
    spans = sorted(spans, key=lambda x: x[0])

    for (start, end) in spans:

        span = spacy_doc[start:end].text
        span = span.title() if span.isupper() else span
        last_token = spacy_doc[end - 1].text

        if span in COUNTRIES:
            yield start, end, "GPE"

        if end <= (start + 3) and (span in NORPS or last_token in NORPS or last_token.rstrip("s") in NORPS):
            yield start, end, "NORP"

        if span in LANGUAGES and spacy_doc[start].tag_ == "NNP":
            yield start, end, "LANGUAGE"

        if last_token in FACILITIES and end > start + 1:
            yield start, end, "FAC"

        if last_token in EVENTS and end > start + 1:
            yield start, end, "EVENT"


class SpanConstraintAnnotator(BaseAnnotator):
    """Annotation by looking at text spans (from another lb_source) that satisfy a span-level constratint"""

    def __init__(self, constraint, initial_source_name, prefix):

        super(SpanConstraintAnnotator, self).__init__()

        self.constraint = constraint
        self.initial_source_name = initial_source_name
        self.prefix = prefix

    def annotate(self, doc):
        """Annotates one single document"""

        self.clear_source(doc, self.prefix + self.initial_source_name)

        for (start, end), vals in doc.user_data["annotations"][self.initial_source_name].items():
            if self.constraint(doc[start:end]):
                for label, conf in vals:
                    self.add(doc, start, end, label, self.prefix + self.initial_source_name, conf)

        return doc


# ---------------------------------------------------------------- #
# the standardisation of previously annotated labels
class StandardiseAnnotator(BaseAnnotator):
    """Annotator taking existing annotations and standardising them (i.e. changing PER to PERSON,
    or changing LOC to GPE if other annotations suggest the entity is a GPE)"""

    def annotate(self, doc):
        """Annotates one single document"""

        gpe_sources = ["geo_cased", "geo_uncased", "wiki_cased", "wiki_uncased", "core_web_md+c", "doc_majority_cased"]
        company_sources = ["company_type_detector", "crunchbase_cased", "crunchbase_uncased", "doc_majority_cased",
                           "doc_majority_uncased"]

        for source in list(doc.user_data.get("annotations", [])):

            if "unified" in source:
                del doc.user_data["annotations"][source]
                continue
            current_spans = dict(doc.user_data["annotations"][source])
            self.clear_source(doc, source)
            for span, vals in current_spans.items():
                new_vals = []
                for label, conf in vals:
                    if label == "PER":
                        label = "PERSON"
                    if label == "LOC" and (source.startswith("conll")
                                           or source.startswith("BTC")
                                           or source.startswith("SEC")
                                           or source.startswith("doc_majority")):
                        for gpe_source in gpe_sources:
                            if span in doc.user_data["annotations"].get(gpe_source, []):
                                for label2, conf2 in doc.user_data["annotations"][gpe_source][span]:
                                    if label2 == "GPE":
                                        label = "GPE"

                    if label == "ORG" and (source.startswith("conll")
                                           or source.startswith("BTC")
                                           or source.startswith("SEC")
                                           or source.startswith("core_web_md")
                                           or source.startswith("doc_majority")
                                           or "wiki_" in source):
                        for company_source in company_sources:
                            if span in doc.user_data["annotations"].get(company_source, []):
                                for label2, conf2 in doc.user_data["annotations"][company_source][span]:
                                    if label2 == "COMPANY":
                                        label = "COMPANY"
                    new_vals.append((label, conf))
                for label, conf in new_vals:
                    self.add(doc, span[0], span[1], label, source, conf)

        return doc


# ---------------------------------------------------------------- #
# Document-level annotators
class DocumentHistoryAnnotator(BaseAnnotator):
    """Annotation based on the document history:
    1) if a person name has been mentioned in full (at least two consecutive tokens, most often first name followed by
    last name), then mark future occurrences of the last token (last name) as a PERSON as well.
    2) if a company name has been mentioned together with a legal type, mark all other occurrences (possibly without
    the legal type at the end) also as a COMPANY.
    """

    def annotate(self, doc):
        """Annotates one single document"""

        self.clear_source(doc, "doc_history")

        trie = Trie()

        # If the doc has fields, we start with the longest ones (i.e. the content)
        if "fields" in doc.user_data:
            field_lengths = {field: (field_end - field_start) for field, (field_start, field_end) in
                             doc.user_data["fields"].items()}
            sorted_fields = sorted(doc.user_data["fields"].keys(), key=lambda x: field_lengths[x], reverse=True)
            field_boundaries = [doc.user_data["fields"][field_name] for field_name in sorted_fields]
        else:
            field_boundaries = [(0, len(doc))]

        for field_start, field_end in field_boundaries:

            sub_doc = doc[field_start:field_end]
            tokens = tuple(tok.text for tok in sub_doc)

            all_spans = [((start, end), val) for source in doc.user_data["annotations"]
                         for ((start, end), val) in doc.user_data["annotations"][source].items()
                         if source in ["core_web_md+c", "conll2003+c", "full_name_detector", "company_type_detector"]
                         or source.endswith("cased")]

            all_spans = sorted(all_spans, key=lambda x: x[0][0])

            # We search for occurrences of full person names or company names with legal suffix
            for (start, end), val in all_spans:
                if len(val) == 0:
                    continue
                if val[0][0] == "PERSON" and (start + 1) < end < (start + 5):
                    last_name = tokens[end - 1:end]
                    if last_name not in trie:
                        trie[tokens[start:end]] = (start, "PERSON")
                        trie[tokens[end - 1:end]] = (start, "PERSON")

                elif (val[0][0] in {"COMPANY", "ORG"} and (start + 1) < end < (start + 8) and
                      doc[end - 1].lower_.rstrip(".") in LEGAL_SUFFIXES):
                    company_without_suffix = tokens[start:end - 1]
                    if company_without_suffix not in trie:
                        trie[tokens[start:end - 1]] = (start, "COMPANY")
                        trie[tokens[start:end]] = (start, "COMPANY")

            i = 0
            while i < len(tokens):

                span = tokens[i:i + 8]
                prefix_length, prefix_value = trie.longest_prefix(span)

                if prefix_length:
                    initial_offset, label = prefix_value
                    if i > initial_offset:
                        self.add(doc, i, i + prefix_length, label, "doc_history")
                    i += prefix_length
                else:
                    i += 1
        return doc


class DocumentMajorityAnnotator(BaseAnnotator):
    """Annotation based on majority label for the same entity string elsewhere in the document. The
    annotation creates two layers, one for case-sensitive occurrences of the entity string in the document,
    and one for case-insensitive occurrences.
    """

    def annotate(self, doc):
        """Annotates one single document"""

        self.clear_source(doc, "doc_majority_cased")
        self.clear_source(doc, "doc_majority_uncased")

        entity_counts = self.get_counts(doc)

        # And we build a trie to easily search for these entities in the text
        trie = Trie()
        for entity, label_counts in entity_counts.items():

            # We require at least 2 occurences of the text span in the document
            entity_lower = tuple(t.lower() for t in entity)
            nb_occurrences = 0

            tokens_lc = tuple(t.lower_ for t in doc)
            for i in range(len(tokens_lc) - len(entity)):
                if tokens_lc[i:i + len(entity)] == entity_lower:
                    nb_occurrences += 1

            # We select the majority label (and give a small preference to rarer/more detailed labels)
            majority_label = max(label_counts, key=lambda x: (label_counts.get(x) * 1000 +
                                                              (1 if x in {"PRODUCT", "COMPANY"} else 0)))
            if nb_occurrences > 1:
                trie[entity] = majority_label

        # Searching for case-sensitive occurrences of the entities
        self.add_annotations(doc, trie)
        self.add_annotations(doc, trie, False)

        return doc

    @staticmethod
    def get_counts(doc):

        entity_counts = {}

        # We first count the possible labels for each span
        span_labels = {}

        sources = ['BTC', 'BTC+c', 'company_type_detector', 'conll2003', 'conll2003+c', 'core_web_md', 'core_web_md+c',
                   'crunchbase_cased', 'crunchbase_uncased', 'date_detector', 'doc_history', 'full_name_detector',
                   'geo_cased',
                   'geo_uncased', 'legal_detector', 'misc_detector', 'money_detector', 'number_detector',
                   'product_cased',
                   'product_uncased', 'snips', 'time_detector', 'wiki_cased', 'wiki_small_cased']

        for source in sources:

            if source not in doc.user_data["annotations"]:
                continue

            for (start, end), vals in doc.user_data["annotations"][source].items():

                if (start, end) not in span_labels:
                    span_labels[(start, end)] = {}

                for label, conf in vals:
                    span_labels[(start, end)][label] = span_labels[(start, end)].get(label, 0) + conf

                # We also look at overlapping spans (weighted by their overlap ratio)
                for start2, end2, vals2 in get_overlaps(start, end, doc.user_data["annotations"], sources):
                    if (start, end) != (start2, end2):
                        overlap = (min(end, end2) - max(start, start2)) / (end - start)
                        for label2, conf2 in vals2:
                            span_labels[(start, end)][label2] = span_labels[(start, end)].get(label2,
                                                                                              0) + conf2 * overlap

                            # We normalise
        for (start, end), label_counts in list(span_labels.items()):
            span_labels[(start, end)] = {label: count / sum(label_counts.values())
                                         for label, count in label_counts.items()}

        # We then count the label occurrences per entity string
        tokens = tuple(tok.text for tok in doc)
        for (start, end), weighted_labels in span_labels.items():
            span_string = tokens[start:end]
            if span_string in entity_counts:
                for label, label_weight in weighted_labels.items():
                    entity_counts[span_string][label] = entity_counts[span_string].get(label, 0) + label_weight
            else:
                entity_counts[span_string] = weighted_labels

        return entity_counts

    def add_annotations(self, doc, trie, case_sensitive=True):

        source = "doc_majority_%s" % ("cased" if case_sensitive else "uncased")

        tokens = tuple(tok.text for tok in doc)
        for i in range(len(tokens)):
            span = tokens[i:i + 8]

            prefix_length, label = trie.longest_prefix(span, case_sensitive)

            # We need to check whether the annotation does not overlap with itself
            if label:
                is_compatible = True
                for (start2, end2, label2) in get_overlaps(i, i + prefix_length, doc.user_data["annotations"],
                                                           [source]):

                    # If an overlap is detected, we select the longest span
                    if end2 - start2 < prefix_length:
                        del doc.user_data["annotations"][source][(start2, end2)]
                    else:
                        is_compatible = False
                        break
                if is_compatible:
                    self.add(doc, i, i + prefix_length, label, source)


# ---------------------------------------------------------------- #
# document construction functions
def set_custom_boundaries(doc, start_indices):
    """
    helper function
    :param doc: spaCy document
    :param start_indices: indicates the start of a sentence
    :return: updated document
    """
    for i in range(len(doc)):
        doc[i].is_sent_start = False
    doc[0].is_sent_start = True
    for i in start_indices[:-1]:
        doc[i].is_sent_start = True
    return doc


def construct_doc(sents: List[List[str]], spacy_model):
    # construct sentences
    s_tokens = []
    for s in sents:
        s_tokens += s
    sent = ' '.join([' '.join(s) for s in sents])
    sep_sent = ' '.join([' '.join(s + ['@SB@']) for s in sents])

    # helper document
    if 'set_boundary_with_sep' in spacy_model.pipe_names:
        spacy_model.remove_pipe('set_boundary_with_sep')
    sep_doc = spacy_model(sep_sent)

    n_sep = 0
    start_indices = list()
    for i, token in enumerate(sep_doc):
        if token.text == '@SB@':
            n_sep += 1
            start_indices.append(i + 1 - n_sep)

    def set_boundary_with_sep(x):
        return set_custom_boundaries(x, start_indices)

    spacy_model.add_pipe(set_boundary_with_sep, before="parser")
    doc = spacy_model(sent)
    return doc


# ---------------------------------------------------------------- #
# test files

def test_2(doc):
    annotator = GazetteerAnnotator(WIKIDATA, "wiki")

    annotator.annotate(doc)
    print()


if __name__ == '__main__':
    news_text = """ATLANTA  (Reuters) - Retailer Best Buy Co, seeking new ways to appeal to cost-conscious shoppers, 
    said on Tuesday it is selling refurbished versions of Apple Inc's iPhone 3G at its stores that are priced about 
    $50 less than new iPhones. The electronics chain said the used iPhones, which were returned within 30 days of 
    purchase, are priced at $149 for the model with 8 gigabytes of storage, while the 16-gigabyte version is $249. A 
    two-year service contract with AT&T Inc is required. New iPhone 3Gs currently sell for $199 and $299 at Best Buy 
    Mobile stores. "This is focusing on customers' needs, trying to provide as wide a range of products and networks 
    for our consumers," said Scott Moore, vice president of marketing for Best Buy Mobile. Buyers of first-generation 
    iPhones can also upgrade to the faster refurbished 3G models at Best Buy, he said. Moore said AT&T, the exclusive 
    wireless provider for the iPhone, offers refurbished iPhones online. The sale of used iPhones comes as Best Buy, 
    the top consumer electronics chain, seeks ways to fend off increased competition from discounters such as 
    Wal-Mart Stores Inc, which began selling the popular phone late last month. Wal-Mart sells a new 8-gigabyte 
    iPhone 3G for $197 and $297 for the 16-gigabyte model. The iPhone is also sold at Apple stores and AT&T stores. 
    Moore said Best Buy's move was not in response to other retailers' actions. (Reporting by  Karen Jacobs ; Editing 
    by  Andre Grenon ) """

    # substitute multiple spaces by one
    news_text = re.sub(r'\s+', ' ', news_text)
    nlp = spacy.load("en_core_web_md")
    document = nlp(news_text)

    test_2(document)
    # test_3()
