"""
--------------------------------------------------------------------------------
List of Labels, annotation formats, and
prior probabilities used for source selection and model initialization
"""
import numpy as np
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.absolute())

OntoNotes_LABELS = ['CARDINAL', "COMPANY", 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY',
                    'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

CoNLL_LABELS = ["PER", "LOC", "ORG", "MISC"]

OntoNotes_BIO = ["O"] + ["%s-%s" % (bi, label) for label in OntoNotes_LABELS for bi in "BI"]
OntoNotes_INDICES = {label: i for i, label in enumerate(OntoNotes_BIO)}

CoNLL_BIO = ["O"] + ["%s-%s" % (bi, label) for label in CoNLL_LABELS for bi in "BI"]
CoNLL_INDICES = {label: i for i, label in enumerate(CoNLL_BIO)}

CoNLL_SOURCE_NAMES = ['BTC', 'BTC+c', 'SEC', 'SEC+c', 'company_type_detector', 'compound_detector',
                      'conll2003', 'conll2003+c', 'core_web_md', 'core_web_md+c', 'crunchbase_cased',
                      'crunchbase_uncased',
                      'date_detector',
                      'doc_history', 'doc_majority_cased', 'doc_majority_uncased', 'full_name_detector', 'geo_cased',
                      'geo_uncased',
                      'infrequent_compound_detector', 'infrequent_nnp_detector', 'infrequent_proper2_detector',
                      'infrequent_proper_detector',
                      'legal_detector', 'misc_detector', 'money_detector',
                      'multitoken_crunchbase_cased', 'multitoken_crunchbase_uncased', 'multitoken_geo_cased',
                      'multitoken_geo_uncased',
                      'multitoken_product_cased', 'multitoken_product_uncased', 'multitoken_wiki_cased',
                      'multitoken_wiki_small_cased',
                      'multitoken_wiki_small_uncased', 'multitoken_wiki_uncased', 'nnp_detector', 'number_detector',
                      'product_cased',
                      'product_uncased', 'proper2_detector', 'proper_detector', 'snips', 'time_detector', 'wiki_cased',
                      'wiki_small_cased',
                      'wiki_small_uncased', 'wiki_uncased']

CoNLL_SOURCE_TO_KEEP = ['BTC+c', 'SEC+c', 'core_web_md+c', 'crunchbase_cased', 'crunchbase_uncased',
                        'doc_majority_cased', 'doc_majority_uncased',
                        'full_name_detector', 'geo_cased', 'geo_uncased', 'misc_detector',
                        'wiki_cased', 'wiki_uncased']


NUMBER_NERS = ["CARDINAL", "DATE", "MONEY", "ORDINAL", "PERCENT", "QUANTITY", "TIME"]

# the numbers are precision and recall
CoNLL_SOURCE_PRIORS = {
    'BTC': {lbs: (0.4, 0.4) if lbs in ["COMPANY", "ORG", "PERSON", "GPE", "LOC"] else (0.3, 0.3) for lbs in
            OntoNotes_LABELS if
            lbs not in NUMBER_NERS},
    'BTC+c': {lbs: (0.5, 0.5) if lbs in ["COMPANY", "ORG", "PERSON", "GPE", "LOC", "MONEY"] else (0.4, 0.4) for lbs in
              OntoNotes_LABELS},
    'SEC': {lbs: (0.1, 0.1) if lbs in ["COMPANY", "ORG", "PERSON", "GPE", "LOC"] else (0.05, 0.05) for lbs in
            OntoNotes_LABELS if
            lbs not in NUMBER_NERS},
    'SEC+c': {lbs: (0.1, 0.1) if lbs in ["COMPANY", "ORG", "PERSON", "GPE", "LOC", "MONEY"] else (0.05, 0.05) for lbs in
              OntoNotes_LABELS},
    'company_type_detector': {'COMPANY': (0.9999, 0.4)},
    'compound_detector': {lbs: (0.7, 0.8) if lbs not in NUMBER_NERS else (0.01, 0.01) for lbs in OntoNotes_LABELS},
    'conll2003': {lbs: (0.7, 0.7) if lbs in ["COMPANY", "ORG", "PERSON", "GPE", "LOC"] else (0.4, 0.4)
                  for lbs in OntoNotes_LABELS if lbs not in NUMBER_NERS},
    'conll2003+c': {lbs: (0.7, 0.7) if lbs in ["COMPANY", "ORG", "PERSON", "GPE", "LOC"] else (0.4, 0.4)
                    for lbs in OntoNotes_LABELS},
    "core_web_md": {lbs: (0.9, 0.9) for lbs in OntoNotes_LABELS},
    "core_web_md+c": {lbs: (0.95, 0.95) for lbs in OntoNotes_LABELS},
    "crunchbase_cased": {lbs: (0.7, 0.6) for lbs in ["PERSON", "ORG", "COMPANY"]},
    "crunchbase_uncased": {lbs: (0.6, 0.7) for lbs in ["PERSON", "ORG", "COMPANY"]},
    'date_detector': {'DATE': (0.9, 0.9)},
    'doc_history': {lbs: (0.99, 0.4) for lbs in ["PERSON", "COMPANY"]},
    'doc_majority_cased': {lbs: (0.98, 0.4) for lbs in OntoNotes_LABELS},
    'doc_majority_uncased': {lbs: (0.95, 0.5) for lbs in OntoNotes_LABELS},
    'full_name_detector': {'PERSON': (0.9999, 0.4)},
    "geo_cased": {lbs: (0.8, 0.8) for lbs in ["GPE", "LOC"]},
    "geo_uncased": {lbs: (0.8, 0.8) for lbs in ["GPE", "LOC"]},
    'infrequent_compound_detector': {lbs: (0.7, 0.8) if lbs not in NUMBER_NERS else (0.01, 0.01) for lbs in
                                     OntoNotes_LABELS},
    'infrequent_nnp_detector': {lbs: (0.7, 0.8) if lbs not in NUMBER_NERS else (0.01, 0.01) for lbs in
                                OntoNotes_LABELS},
    'infrequent_proper2_detector': {lbs: (0.7, 0.8) if lbs not in NUMBER_NERS else (0.01, 0.01) for lbs in
                                    OntoNotes_LABELS},
    'infrequent_proper_detector': {lbs: (0.7, 0.8) if lbs not in NUMBER_NERS else (0.01, 0.01) for lbs in
                                   OntoNotes_LABELS},
    'legal_detector': {"LAW": (0.8, 0.8)},
    'misc_detector': {lbs: (0.7, 0.7) for lbs in ["NORP", "EVENT", "FAC", "GPE", "LANGUAGE"]},
    'money_detector': {'MONEY': (0.9, 0.9)},
    'multitoken_crunchbase_cased': {lbs: (0.8, 0.6) for lbs in ["PERSON", "ORG", "COMPANY"]},
    'multitoken_crunchbase_uncased': {lbs: (0.7, 0.7) for lbs in ["PERSON", "ORG", "COMPANY"]},
    'multitoken_geo_cased': {lbs: (0.8, 0.6) for lbs in ["GPE", "LOC"]},
    'multitoken_geo_uncased': {lbs: (0.7, 0.7) for lbs in ["GPE", "LOC"]},
    'multitoken_product_cased': {"PRODUCT": (0.8, 0.6)},
    'multitoken_product_uncased': {"PRODUCT": (0.7, 0.7)},
    'multitoken_wiki_cased': {lbs: (0.8, 0.6) for lbs in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
    'multitoken_wiki_small_cased': {lbs: (0.8, 0.6) for lbs in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
    'multitoken_wiki_small_uncased': {lbs: (0.7, 0.7) for lbs in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
    'multitoken_wiki_uncased': {lbs: (0.7, 0.7) for lbs in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
    'nnp_detector': {lbs: (0.8, 0.8) if lbs not in NUMBER_NERS else (0.01, 0.01) for lbs in OntoNotes_LABELS},
    "number_detector": {lbs: (0.9, 0.9) for lbs in ["CARDINAL", "ORDINAL", "QUANTITY", "PERCENT"]},
    'product_cased': {"PRODUCT": (0.7, 0.6)},
    'product_uncased': {"PRODUCT": (0.6, 0.7)},
    'proper2_detector': {lbs: (0.6, 0.8) if lbs not in NUMBER_NERS else (0.01, 0.01) for lbs in OntoNotes_LABELS},
    'proper_detector': {lbs: (0.6, 0.8) if lbs not in NUMBER_NERS else (0.01, 0.01) for lbs in OntoNotes_LABELS},
    "snips": {lbs: (0.8, 0.8) for lbs in ["DATE", "TIME", "PERCENT", "CARDINAL", "ORDINAL", "MONEY"]},
    'time_detector': {'TIME': (0.9, 0.9)},
    'wiki_cased': {lbs: (0.6, 0.5) for lbs in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
    'wiki_small_cased': {lbs: (0.7, 0.6) for lbs in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
    'wiki_small_uncased': {lbs: (0.6, 0.7) for lbs in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
    'wiki_uncased': {lbs: (0.5, 0.6) for lbs in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]}}

# In some rare cases (due to specialisations of corrections of labels), we also need to add some other labels
for lb_source in ["BTC", "BTC+c", "SEC", "SEC+c", "conll2003", "conll2003+c"]:
    CoNLL_SOURCE_PRIORS[lb_source].update({lbs: (0.8, 0.01) for lbs in NUMBER_NERS})

OUT_RECALL = 0.9
OUT_PRECISION = 0.8


CoNLL_MAPPINGS = {"PERSON": "PER", "COMPANY": "ORG", "GPE": "LOC", 'EVENT': "MISC", 'FAC': "MISC", 'LANGUAGE': "MISC",
                  'LAW': "MISC", 'NORP': "MISC", 'PRODUCT': "MISC", 'WORK_OF_ART': "MISC"}

CONLL_SRC_PRIORS = dict()
for src, priors in CoNLL_SOURCE_PRIORS.items():
    transferred_lbs = dict()
    for lb_name, ps in priors.items():
        norm_lb = CoNLL_MAPPINGS.get(lb_name, lb_name)
        if norm_lb not in CoNLL_LABELS:
            continue
        if norm_lb not in transferred_lbs:
            transferred_lbs[norm_lb] = [list(), list()]
        transferred_lbs[norm_lb][0].append(ps[0])
        transferred_lbs[norm_lb][1].append(ps[1])
    for norm_lb in transferred_lbs:
        transferred_lbs[norm_lb][0] = np.mean(transferred_lbs[norm_lb][0])
        transferred_lbs[norm_lb][1] = np.mean(transferred_lbs[norm_lb][1])
        transferred_lbs[norm_lb] = tuple(transferred_lbs[norm_lb])
    CONLL_SRC_PRIORS[src] = transferred_lbs

NCBI_LABELS = ['DISEASE']
NCBI_BIO = ["O"] + ["%s-%s" % (bi, label) for label in NCBI_LABELS for bi in "BI"]
NCBI_INDICES = {label: i for i, label in enumerate(NCBI_BIO)}

NCBI_SOURCE_NAMES = ['CoreDictionaryUncased', 'CoreDictionaryExact', 'CancerLike',
                     'CommonSuffixes', 'Deficiency', 'Disorder',
                     'Lesion', 'Syndrome', 'BodyTerms',
                     'OtherPOS', 'StopWords', 'Punctuation',
                     'PossessivePhrase', 'HyphenatedPhrase', 'ElmoLinkingRule',
                     'CommonBigram', 'ExtractedPhrase']

NCBI_SOURCES_TO_KEEP = ['CoreDictionaryUncased', 'CoreDictionaryExact', 'CancerLike', 'BodyTerms', 'ExtractedPhrase']
# NCBI_SOURCES_TO_KEEP = ['CoreDictionaryUncased', 'CoreDictionaryExact']

NCBI_SOURCE_PRIORS = {
    'CoreDictionaryUncased': {lbs: (0.8, 0.7) for lbs in NCBI_LABELS},
    'CoreDictionaryExact': {lbs: (0.9, 0.5) for lbs in NCBI_LABELS},
    'CancerLike': {lbs: (0.5, 0.4) for lbs in NCBI_LABELS},
    'CommonSuffixes': {'DISEASE': (0.1, 0.1)},
    'Deficiency': {'DISEASE': (0.1, 0.1)},
    'Disorder': {'DISEASE': (0.1, 0.1)},
    'Lesion': {'DISEASE': (0.1, 0.1)},
    'Syndrome': {'DISEASE': (0.1, 0.1)},
    "BodyTerms": {lbs: (0.5, 0.4) for lbs in NCBI_LABELS},
    "OtherPOS": {'DISEASE': (0.1, 0.1)},
    "StopWords": {'DISEASE': (0.1, 0.1)},
    "Punctuation": {'DISEASE': (0.1, 0.1)},
    "PossessivePhrase": {'DISEASE': (0.1, 0.1)},
    "HyphenatedPhrase": {'DISEASE': (0.1, 0.1)},
    'ElmoLinkingRule': {'DISEASE': (0.1, 0.1)},
    'CommonBigram': {'DISEASE': (0.1, 0.1)},
    'ExtractedPhrase': {'DISEASE': (0.9, 0.9)}
}

LAPTOP_LABELS = ['TERM']
LAPTOP_BIO = ["O"] + ["%s-%s" % (bi, label) for label in LAPTOP_LABELS for bi in "BI"]
LAPTOP_INDICES = {label: i for i, label in enumerate(LAPTOP_BIO)}

LAPTOP_SOURCE_NAMES = ['CoreDictionary', 'OtherTerms', 'ReplaceThe', 'iStuff',
                       'Feelings', 'ProblemWithThe', 'External', 'StopWords',
                       'Punctuation', 'Pronouns', 'NotFeatures', 'Adv',
                       'CompoundPhrase', 'ElmoLinkingRule', 'ExtractedPhrase', 'ConsecutiveCapitals']

LAPTOP_SOURCES_TO_KEEP = ['CoreDictionary', 'OtherTerms', 'iStuff', 'ExtractedPhrase', 'ConsecutiveCapitals']

LAPTOP_SOURCE_PRIORS = {
    'CoreDictionary': {lbs: (0.9, 0.7) for lbs in LAPTOP_LABELS},
    'OtherTerms': {lbs: (0.6, 0.5) for lbs in LAPTOP_LABELS},
    'ReplaceThe': {lbs: (0.1, 0.1) for lbs in LAPTOP_LABELS},
    'iStuff': {'TERM': (0.6, 0.4)},
    'Feelings': {'TERM': (0.1, 0.1)},
    'ProblemWithThe': {'TERM': (0.1, 0.1)},
    'External': {'TERM': (0.5, 0.4)},
    'StopWords': {'TERM': (0.1, 0.1)},
    "Punctuation": {lbs: (0.5, 0.4) for lbs in LAPTOP_LABELS},
    "Pronouns": {'TERM': (0.1, 0.1)},
    "NotFeatures": {'TERM': (0.1, 0.1)},
    "Adv": {'TERM': (0.1, 0.1)},
    "CompoundPhrase": {'TERM': (0.1, 0.1)},
    "ElmoLinkingRule": {'TERM': (0.6, 0.4)},
    'ExtractedPhrase': {'TERM': (0.9, 0.9)},
    'ConsecutiveCapitals': {'TERM': (0.7, 0.6)}
}

BC5CDR_LABELS = ['Chemical', 'Disease']
BC5CDR_BIO = ["O"] + ["%s-%s" % (bi, label) for label in BC5CDR_LABELS for bi in "BI"]
BC5CDR_INDICES = {label: i for i, label in enumerate(BC5CDR_BIO)}

BC5CDR_SOURCE_NAMES = [
    'DictCore-Chemical', 'DictCore-Chemical-Exact', 'DictCore-Disease', 'DictCore-Disease-Exact',
    'Element, Ion, or Isotope', 'Organic Chemical', 'Antibiotic', 'Disease or Syndrome',
    'BodyTerms', 'Acronyms', 'Damage', 'Disease',
    'Disorder', 'Lesion', 'Syndrome', 'ChemicalSuffixes',
    'CancerLike', 'DiseaseSuffixes', 'DiseasePrefixes', 'Induced',
    'Vitamin', 'Acid', 'OtherPOS', 'StopWords',
    'CommonOther', 'Punctuation', 'PossessivePhrase', 'HyphenatedPrefix',
    'PostHyphen', 'ExtractedPhrase'
]

BC5CDR_SOURCES_TO_KEEP = [
    'DictCore-Chemical', 'DictCore-Chemical-Exact', 'DictCore-Disease', 'DictCore-Disease-Exact',
    'Organic Chemical', 'Disease or Syndrome', 'PostHyphen', 'ExtractedPhrase'
]

BC5CDR_SOURCE_PRIORS = {
    'DictCore-Chemical': {'Chemical': (0.9, 0.9), 'Disease': (0.1, 0.1)},
    'DictCore-Chemical-Exact': {'Chemical': (0.9, 0.5), 'Disease': (0.1, 0.1)},
    'DictCore-Disease': {'Chemical': (0.1, 0.1), 'Disease': (0.9, 0.9)},
    'DictCore-Disease-Exact': {'Chemical': (0.1, 0.1), 'Disease': (0.9, 0.5)},
    'Element, Ion, or Isotope': {'Chemical': (0.9, 0.4), 'Disease': (0.1, 0.1)},
    'Organic Chemical': {'Chemical': (0.9, 0.9), 'Disease': (0.1, 0.1)},
    'Antibiotic': {'Chemical': (0.9, 0.4), 'Disease': (0.1, 0.1)},
    'Disease or Syndrome': {'Chemical': (0.1, 0.1), 'Disease': (0.9, 0.7)},
    'BodyTerms': {'Chemical': (0.1, 0.1), 'Disease': (0.7, 0.3)},
    'Acronyms': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'Damage': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'Disease': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'Disorder': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'Lesion': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'Syndrome': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'ChemicalSuffixes': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'CancerLike': {'Chemical': (0.1, 0.1), 'Disease': (0.7, 0.3)},
    'DiseaseSuffixes': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'DiseasePrefixes': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'Induced': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'Vitamin': {'Chemical': (0.9, 0.3), 'Disease': (0.1, 0.1)},
    'Acid': {'Chemical': (0.9, 0.3), 'Disease': (0.1, 0.1)},
    'OtherPOS': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'StopWords': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'CommonOther': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'Punctuation': {'Chemical': (0.1, 0.1), 'Disease': (0.1, 0.1)},
    'PossessivePhrase': {'Chemical': (0.2, 0.2), 'Disease': (0.2, 0.2)},
    'HyphenatedPrefix': {'Chemical': (0.2, 0.2), 'Disease': (0.2, 0.2)},
    'PostHyphen': {'Chemical': (0.8, 0.3), 'Disease': (0.8, 0.3)},
    'ExtractedPhrase': {'Chemical': (0.8, 0.3), 'Disease': (0.8, 0.3)},
}
