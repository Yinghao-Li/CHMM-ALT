from .wiser.rules import TaggingRule, LinkingRule, DictionaryMatcher, ElmoLinkingRule


def bc5cdr_annotators(docs, reader):
    dict_core_chem = set()
    dict_core_chem_exact = set()
    dict_core_dis = set()
    dict_core_dis_exact = set()

    # Tagging rules
    with open('../Dependency/AutoNER_dicts/BC5CDR/dict_core.txt') as f:
        for line in f.readlines():
            line = line.strip().split(None, 1)
            entity_type = line[0]
            tokens = reader.get_tokenizer()(line[1])
            term = tuple([str(x) for x in tokens])

            if len(term) > 1 or len(term[0]) > 3:
                if entity_type == 'Chemical':
                    dict_core_chem.add(term)
                elif entity_type == 'Disease':
                    dict_core_dis.add(term)
                else:
                    raise Exception()
            else:
                if entity_type == 'Chemical':
                    dict_core_chem_exact.add(term)
                elif entity_type == 'Disease':
                    dict_core_dis_exact.add(term)
                else:
                    raise Exception()

    lf = DictionaryMatcher(
        "DictCore-Chemical",
        dict_core_chem,
        i_label="I-Chemical",
        uncased=True)
    lf.apply(docs)
    lf = DictionaryMatcher(
        "DictCore-Chemical-Exact",
        dict_core_chem_exact,
        i_label="I-Chemical",
        uncased=False)
    lf.apply(docs)
    lf = DictionaryMatcher(
        "DictCore-Disease",
        dict_core_dis,
        i_label="I-Disease",
        uncased=True)
    lf.apply(docs)
    lf = DictionaryMatcher(
        "DictCore-Disease-Exact",
        dict_core_dis_exact,
        i_label="I-Disease",
        uncased=False)
    lf.apply(docs)

    terms = []
    with open('../Dependency/umls/umls_element_ion_or_isotope.txt', 'r') as f:
        for line in f.readlines():
            terms.append(line.strip().split(" "))
    lf = DictionaryMatcher(
        "Element, Ion, or Isotope",
        terms,
        i_label='I-Chemical',
        uncased=True,
        match_lemmas=True)
    lf.apply(docs)

    terms = []
    with open('../Dependency/umls/umls_organic_chemical.txt', 'r') as f:
        for line in f.readlines():
            terms.append(line.strip().split(" "))
    lf = DictionaryMatcher(
        "Organic Chemical",
        terms,
        i_label='I-Chemical',
        uncased=True,
        match_lemmas=True)
    lf.apply(docs)

    terms = []
    with open('../Dependency/umls/umls_antibiotic.txt', 'r') as f:
        for line in f.readlines():
            terms.append(line.strip().split(" "))
    lf = DictionaryMatcher(
        "Antibiotic",
        terms,
        i_label='I-Chemical',
        uncased=True,
        match_lemmas=True)
    lf.apply(docs)

    terms = []
    with open('../Dependency/umls/umls_disease_or_syndrome.txt', 'r') as f:
        for line in f.readlines():
            terms.append(line.strip().split(" "))
    lf = DictionaryMatcher(
        "Disease or Syndrome",
        terms,
        i_label='I-Disease',
        uncased=True,
        match_lemmas=True)
    lf.apply(docs)

    terms = []
    with open('../Dependency/umls/umls_body_part.txt', 'r') as f:
        for line in f.readlines():
            terms.append(line.strip().split(" "))
    lf = DictionaryMatcher(
        "TEMP",
        terms,
        i_label='TEMP',
        uncased=True,
        match_lemmas=True)
    lf.apply(docs)

    # noinspection PyShadowingNames
    class BodyTerms(TaggingRule):
        def apply_instance(self, instance):
            tokens = [token.text.lower() for token in instance['tokens']]
            labels = ['ABS'] * len(tokens)

            terms = {"cancer", "cancers", "damage", "disease", "diseases", "pain", "injury", "injuries"}

            for i in range(0, len(tokens) - 1):
                if instance['WISER_LABELS']['TEMP'][i] == 'TEMP':
                    if tokens[i + 1] in terms:
                        labels[i] = "I-Disease"
                        labels[i + 1] = "I-Disease"
            return labels

    lf = BodyTerms()
    lf.apply(docs)

    for doc in docs:
        del doc['WISER_LABELS']['TEMP']

    class Acronyms(TaggingRule):
        other_lfs = {
            'I-Chemical': ("Antibiotic", "Element, Ion, or Isotope", "Organic Chemical"),
            'I-Disease': ("BodyTerms", "Disease or Syndrome")
        }

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            active = False
            for tag, lf_names in self.other_lfs.items():
                acronyms = set()
                for lf_name in lf_names:
                    for i in range(len(instance['tokens']) - 2):
                        if instance['WISER_LABELS'][lf_name][i] == tag:
                            active = True
                        elif active and instance['tokens'][i].text == '(' and \
                                instance['tokens'][i + 2].pos_ == "PUNCT" and \
                                instance['tokens'][i + 1].pos_ != "NUM":
                            acronyms.add(instance['tokens'][i + 1].text)
                            active = False
                        else:
                            active = False

                for i, token in enumerate(instance['tokens']):
                    if token.text in acronyms:
                        labels[i] = tag

            return labels

    lf = Acronyms()
    lf.apply(docs)

    class Damage(TaggingRule):

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens']) - 1):
                if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                                   1].lemma_ == 'damage':
                    labels[i] = 'I-Disease'
                    labels[i + 1] = 'I-Disease'

                    # Adds any other compound tokens before the phrase
                    for j in range(i - 1, -1, -1):
                        if instance['tokens'][j].dep_ == 'compound':
                            labels[j] = 'I-Disease'
                        else:
                            break

            return labels

    lf = Damage()
    lf.apply(docs)

    class Disease(TaggingRule):

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens']) - 1):
                if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                                   1].lemma_ == 'disease':
                    labels[i] = 'I-Disease'
                    labels[i + 1] = 'I-Disease'

                    # Adds any other compound tokens before the phrase
                    for j in range(i - 1, -1, -1):
                        if instance['tokens'][j].dep_ == 'compound':
                            labels[j] = 'I-Disease'
                        else:
                            break

            return labels

    lf = Disease()
    lf.apply(docs)

    class Disorder(TaggingRule):

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens']) - 1):
                if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                                   1].lemma_ == 'disorder':
                    labels[i] = 'I-Disease'
                    labels[i + 1] = 'I-Disease'

                    # Adds any other compound tokens before the phrase
                    for j in range(i - 1, -1, -1):
                        if instance['tokens'][j].dep_ == 'compound':
                            labels[j] = 'I-Disease'
                        else:
                            break

            return labels

    lf = Disorder()
    lf.apply(docs)

    class Lesion(TaggingRule):

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens']) - 1):
                if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                                   1].lemma_ == 'lesion':
                    labels[i] = 'I-Disease'
                    labels[i + 1] = 'I-Disease'

                    # Adds any other compound tokens before the phrase
                    for j in range(i - 1, -1, -1):
                        if instance['tokens'][j].dep_ == 'compound':
                            labels[j] = 'I-Disease'
                        else:
                            break

            return labels

    lf = Lesion()
    lf.apply(docs)

    class Syndrome(TaggingRule):

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens']) - 1):
                if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                                   1].lemma_ == 'syndrome':
                    labels[i] = 'I-Disease'
                    labels[i + 1] = 'I-Disease'

                    # Adds any other compound tokens before the phrase
                    for j in range(i - 1, -1, -1):
                        if instance['tokens'][j].dep_ == 'compound':
                            labels[j] = 'I-Disease'
                        else:
                            break

            return labels

    lf = Syndrome()
    lf.apply(docs)

    exceptions = {'determine', 'baseline', 'decline',
                  'examine', 'pontine', 'vaccine',
                  'routine', 'crystalline', 'migraine',
                  'alkaline', 'midline', 'borderline',
                  'cocaine', 'medicine', 'medline',
                  'asystole', 'control', 'protocol',
                  'alcohol', 'aerosol', 'peptide',
                  'provide', 'outside', 'intestine',
                  'combine', 'delirium', 'VIP'}

    suffixes = ('ine', 'ole', 'ol', 'ide', 'ine', 'ium', 'epam')

    class ChemicalSuffixes(TaggingRule):
        def apply_instance(self, instance):

            labels = ['ABS'] * len(instance['tokens'])

            acronyms = set()
            for i, t in enumerate(instance['tokens']):
                if len(t.lemma_) >= 7 and t.lemma_ not in exceptions and t.lemma_.endswith(
                        suffixes):
                    labels[i] = 'I-Chemical'

                    if i < len(instance['tokens']) - 3 and \
                            instance['tokens'][i + 1].text == '(' and \
                            instance['tokens'][i + 3].text == ')':
                        acronyms.add(instance['tokens'][i + 2].text)

            for i, t in enumerate(instance['tokens']):
                if t.text in acronyms and t.text not in exceptions:
                    labels[i] = 'I-Chemical'
            return labels

    lf = ChemicalSuffixes()
    lf.apply(docs)

    # noinspection PyShadowingNames
    class CancerLike(TaggingRule):
        def apply_instance(self, instance):
            tokens = [token.text.lower() for token in instance['tokens']]
            labels = ['ABS'] * len(tokens)

            suffixes = ("edema", "toma", "coma", "noma")

            for i, token in enumerate(tokens):
                for suffix in suffixes:
                    if token.endswith(suffix) or token.endswith(suffix + "s"):
                        labels[i] = 'I-Disease'
            return labels

    lf = CancerLike()
    lf.apply(docs)

    exceptions = {'diagnosis', 'apoptosis', 'prognosis', 'metabolism'}

    suffixes = ("agia", "cardia", "trophy", "itis",
                "emia", "enia", "pathy", "plasia", "lism", "osis")

    class DiseaseSuffixes(TaggingRule):
        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i, t in enumerate(instance['tokens']):
                if len(t.lemma_) >= 5 and t.lemma_.lower(
                ) not in exceptions and t.lemma_.endswith(suffixes):
                    labels[i] = 'I-Disease'

            return labels

    lf = DiseaseSuffixes()
    lf.apply(docs)

    exceptions = {'hypothesis', 'hypothesize', 'hypobaric', 'hyperbaric'}

    prefixes = ('hyper', 'hypo')

    class DiseasePrefixes(TaggingRule):
        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i, t in enumerate(instance['tokens']):
                if len(t.lemma_) >= 5 and t.lemma_.lower(
                ) not in exceptions and t.lemma_.startswith(prefixes):
                    if instance['tokens'][i].pos_ == "NOUN":
                        labels[i] = 'I-Disease'

            return labels

    lf = DiseasePrefixes()
    lf.apply(docs)

    exceptions = {
        "drug",
        "pre",
        "therapy",
        "anesthetia",
        "anesthetic",
        "neuroleptic",
        "saline",
        "stimulus"}

    class Induced(TaggingRule):
        def apply_instance(self, instance):

            labels = ['ABS'] * len(instance['tokens'])

            for i in range(1, len(instance['tokens']) - 3):
                if instance['tokens'][i].text == '-' and \
                        instance['tokens'][i + 1].lemma_ == 'induce':
                    labels[i] = 'O'
                    labels[i + 1] = 'O'
                    if instance['tokens'][i - 1].lemma_ in exceptions or \
                            instance['tokens'][i - 1].pos_ == "PUNCT":
                        labels[i - 1] = 'O'
                    else:
                        labels[i - 1] = 'I-Chemical'
            return labels

    lf = Induced()
    lf.apply(docs)

    class Vitamin(TaggingRule):
        def apply_instance(self, instance):

            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens']) - 1):
                if instance['tokens'][i].text.lower() == 'vitamin':
                    labels[i] = 'I-Chemical'
                    if len(instance['tokens'][i + 1].text) <= 2 and \
                            instance['tokens'][i + 1].text.isupper():
                        labels[i + 1] = 'I-Chemical'

            return labels

    lf = Vitamin()
    lf.apply(docs)

    # noinspection PyShadowingNames
    class Acid(TaggingRule):
        def apply_instance(self, instance):

            labels = ['ABS'] * len(instance['tokens'])
            tokens = instance['tokens']

            for i, t in enumerate(tokens):
                if i > 0 and t.text.lower(
                ) == 'acid' and tokens[i - 1].text.endswith('ic'):
                    labels[i] = 'I-Chemical'
                    labels[i - 1] = 'I-Chemical'

            return labels

    lf = Acid()
    lf.apply(docs)

    class OtherPOS(TaggingRule):
        other_pos = {"ADP", "ADV", "DET", "VERB"}

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(0, len(instance['tokens'])):
                # Some chemicals with long names get tagged as verbs
                if instance['tokens'][i].pos_ in self.other_pos and \
                        instance['WISER_LABELS']['Organic Chemical'][i] == 'ABS' and \
                        instance['WISER_LABELS']['DictCore-Chemical'][i] == 'ABS':
                    labels[i] = "O"
            return labels

    lf = OtherPOS()
    lf.apply(docs)

    stop_words = {"a", "an", "as", "be", "but", "do", "even",
                  "for", "from",
                  "had", "has", "have", "i", "in", "is", "its", "just",
                  "may", "my", "no", "not", "on", "or",
                  "than", "that", "the", "these", "this", "those", "to", "very",
                  "what", "which", "who", "with"}

    class StopWords(TaggingRule):

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens'])):
                if instance['tokens'][i].lemma_ in stop_words:
                    labels[i] = 'O'
            return labels

    lf = StopWords()
    lf.apply(docs)

    class CommonOther(TaggingRule):
        other_lemmas = {'patient', '-PRON-', 'induce', 'after', 'study',
                        'rat', 'mg', 'use', 'treatment', 'increase',
                        'day', 'group', 'dose', 'treat', 'case', 'result',
                        'kg', 'control', 'report', 'administration', 'follow',
                        'level', 'suggest', 'develop', 'week', 'compare',
                        'significantly', 'receive', 'mouse',
                        'protein', 'infusion', 'output', 'area', 'effect',
                        'rate', 'weight', 'size', 'time', 'year',
                        'clinical', 'conclusion', 'outcome', 'man', 'woman',
                        'model', 'concentration'}

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])
            for i in range(len(instance['tokens'])):
                if instance['tokens'][i].lemma_ in self.other_lemmas:
                    labels[i] = 'O'
            return labels

    lf = CommonOther()
    lf.apply(docs)

    class Punctuation(TaggingRule):

        other_punc = {"?", "!", ";", ":", ".", ",",
                      "%", "<", ">", "=", "\\"}

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens'])):
                if instance['tokens'][i].text in self.other_punc:
                    labels[i] = 'O'
            return labels

    lf = Punctuation()
    lf.apply(docs)

    # def linking rules
    class PossessivePhrase(LinkingRule):
        def apply_instance(self, instance):
            links = [0] * len(instance['tokens'])
            for i in range(1, len(instance['tokens'])):
                if instance['tokens'][i -
                                      1].text == "'s" or instance['tokens'][i].text == "'s":
                    links[i] = 1

            return links

    lf = PossessivePhrase()
    lf.apply(docs)

    class HyphenatedPrefix(LinkingRule):
        chem_mods = {"alpha", "beta", "gamma", "delta", "epsilon"}

        def apply_instance(self, instance):
            links = [0] * len(instance['tokens'])
            for i in range(1, len(instance['tokens'])):
                if (instance['tokens'][i - 1].text.lower() in self.chem_mods or
                    len(instance['tokens'][i - 1].text) < 2) \
                        and instance['tokens'][i].text == "-":
                    links[i] = 1

            return links

    lf = HyphenatedPrefix()
    lf.apply(docs)

    class PostHyphen(LinkingRule):
        def apply_instance(self, instance):
            links = [0] * len(instance['tokens'])
            for i in range(1, len(instance['tokens'])):
                if instance['tokens'][i - 1].text == "-":
                    links[i] = 1

            return links

    lf = PostHyphen()
    lf.apply(docs)

    dict_full = set()

    with open('../Dependency/AutoNER_dicts/BC5CDR/dict_full.txt') as f:
        for line in f.readlines():
            tokens = reader.get_tokenizer()(line.strip())
            term = tuple([str(x) for x in tokens])
            if len(term) > 1:
                dict_full.add(tuple(term))

    # noinspection PyShadowingNames
    class ExtractedPhrase(LinkingRule):
        def __init__(self, terms):
            self.term_dict = {}

            for term in terms:
                term = [token.lower() for token in term]
                if term[0] not in self.term_dict:
                    self.term_dict[term[0]] = []
                self.term_dict[term[0]].append(term)

            # Sorts the terms in decreasing order so that we match the longest
            # first
            for first_token in self.term_dict.keys():
                to_sort = self.term_dict[first_token]
                self.term_dict[first_token] = sorted(
                    to_sort, reverse=True, key=lambda x: len(x))

        def apply_instance(self, instance):
            tokens = [token.text.lower() for token in instance['tokens']]
            links = [0] * len(instance['tokens'])

            i = 0
            while i < len(tokens):
                if tokens[i] in self.term_dict:
                    candidates = self.term_dict[tokens[i]]
                    for c in candidates:
                        # Checks whether normalized AllenNLP tokens equal the list
                        # of string tokens defining the term in the dictionary
                        if i + len(c) <= len(tokens):
                            equal = True
                            for j in range(len(c)):
                                if tokens[i + j] != c[j]:
                                    equal = False
                                    break

                            # If tokens match, labels the instance tokens
                            if equal:
                                for j in range(i + 1, i + len(c)):
                                    links[j] = 1
                                i = i + len(c) - 1
                                break
                i += 1

            return links

    lf = ExtractedPhrase(dict_full)
    lf.apply(docs)

    return docs


def laptop_annotators(docs):
    dict_core = set()
    with open('../Dependency/AutoNER_dicts/LaptopReview/dict_core.txt') as f:
        for line in f.readlines():
            line = line.strip().split()
            term = tuple(line[1:])
            dict_core.add(term)

    dict_full = set()

    with open('../Dependency/AutoNER_dicts/LaptopReview/dict_full.txt') as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) > 1:
                dict_full.add(tuple(line))

    lf = DictionaryMatcher("CoreDictionary", dict_core, uncased=True, i_label="I")
    lf.apply(docs)

    other_terms = [['BIOS'], ['color'], ['cord'], ['hinge'], ['hinges'],
                   ['port'], ['speaker']]
    lf = DictionaryMatcher("OtherTerms", other_terms, uncased=True, i_label="I")
    lf.apply(docs)

    class ReplaceThe(TaggingRule):
        def apply_instance(self, instance):
            tokens = [token.text for token in instance['tokens']]
            labels = ['ABS'] * len(tokens)

            for i in range(len(tokens) - 2):
                if tokens[i].lower() == 'replace' and tokens[i +
                                                             1].lower() == 'the':
                    if instance['tokens'][i + 2].pos_ == "NOUN":
                        labels[i] = 'O'
                        labels[i + 1] = 'O'
                        labels[i + 2] = 'I'

            return labels

    lf = ReplaceThe()
    lf.apply(docs)

    class iStuff(TaggingRule):
        def apply_instance(self, instance):
            tokens = [token.text for token in instance['tokens']]
            labels = ['ABS'] * len(tokens)

            for i in range(len(tokens)):
                if len(
                        tokens[i]) > 1 and tokens[i][0] == 'i' and tokens[i][1].isupper():
                    labels[i] = 'I'

            return labels

    lf = iStuff()
    lf.apply(docs)

    class Feelings(TaggingRule):
        feeling_words = {"like", "liked", "love", "dislike", "hate"}

        def apply_instance(self, instance):
            tokens = [token.text for token in instance['tokens']]
            labels = ['ABS'] * len(tokens)

            for i in range(len(tokens) - 2):
                if tokens[i].lower() in self.feeling_words and \
                        tokens[i + 1].lower() == 'the':
                    if instance['tokens'][i + 2].pos_ == "NOUN":
                        labels[i] = 'O'
                        labels[i + 1] = 'O'
                        labels[i + 2] = 'I'

            return labels

    lf = Feelings()
    lf.apply(docs)

    class ProblemWithThe(TaggingRule):
        def apply_instance(self, instance):
            tokens = [token.text for token in instance['tokens']]
            labels = ['ABS'] * len(tokens)

            for i in range(len(tokens) - 3):
                if tokens[i].lower() == 'problem' and \
                        tokens[i + 1].lower() == 'with' and tokens[i + 2].lower() == 'the':
                    if instance['tokens'][i + 3].pos_ == "NOUN":
                        labels[i] = 'O'
                        labels[i + 1] = 'O'
                        labels[i + 2] = 'O'
                        labels[i + 3] = 'I'

            return labels

    lf = ProblemWithThe()
    lf.apply(docs)

    class External(TaggingRule):
        def apply_instance(self, instance):
            tokens = [token.text for token in instance['tokens']]
            labels = ['ABS'] * len(tokens)

            for i in range(len(tokens) - 1):
                if tokens[i].lower() == 'external':
                    labels[i] = 'I'
                    labels[i + 1] = 'I'

            return labels

    lf = External()
    lf.apply(docs)

    stop_words = {"a", "and", "as", "be", "but", "do", "even",
                  "for", "from",
                  "had", "has", "have", "i", "in", "is", "its", "just",
                  "my", "no", "not", "of", "on", "or",
                  "that", "the", "these", "this", "those", "to", "very",
                  "what", "which", "who", "with"}

    class StopWords(TaggingRule):

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens'])):
                if instance['tokens'][i].lemma_ in stop_words:
                    labels[i] = 'O'
            return labels

    lf = StopWords()
    lf.apply(docs)

    class Punctuation(TaggingRule):
        pos = {"PUNCT"}

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i, pos in enumerate([token.pos_ for token in instance['tokens']]):
                if pos in self.pos:
                    labels[i] = 'O'

            return labels

    lf = Punctuation()
    lf.apply(docs)

    class Pronouns(TaggingRule):
        pos = {"PRON"}

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i, pos in enumerate([token.pos_ for token in instance['tokens']]):
                if pos in self.pos:
                    labels[i] = 'O'

            return labels

    lf = Pronouns()
    lf.apply(docs)

    class NotFeatures(TaggingRule):
        keywords = {"laptop", "computer", "pc"}

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens'])):
                if instance['tokens'][i].lemma_ in self.keywords:
                    labels[i] = 'O'
            return labels

    lf = NotFeatures()
    lf.apply(docs)

    class Adv(TaggingRule):
        pos = {"ADV"}

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i, pos in enumerate([token.pos_ for token in instance['tokens']]):
                if pos in self.pos:
                    labels[i] = 'O'

            return labels

    lf = Adv()
    lf.apply(docs)

    class CompoundPhrase(LinkingRule):
        def apply_instance(self, instance):
            links = [0] * len(instance['tokens'])
            for i in range(1, len(instance['tokens'])):
                if instance['tokens'][i - 1].dep_ == "compound":
                    links[i] = 1

            return links

    lf = CompoundPhrase()
    lf.apply(docs)

    lf = ElmoLinkingRule(.8)
    lf.apply(docs)

    # noinspection PyShadowingNames
    class ExtractedPhrase(LinkingRule):
        def __init__(self, terms):
            self.term_dict = {}

            for term in terms:
                term = [token.lower() for token in term]
                if term[0] not in self.term_dict:
                    self.term_dict[term[0]] = []
                self.term_dict[term[0]].append(term)

            # Sorts the terms in decreasing order so that we match the longest
            # first
            for first_token in self.term_dict.keys():
                to_sort = self.term_dict[first_token]
                self.term_dict[first_token] = sorted(
                    to_sort, reverse=True, key=lambda x: len(x))

        def apply_instance(self, instance):
            tokens = [token.text.lower() for token in instance['tokens']]
            links = [0] * len(instance['tokens'])

            i = 0
            while i < len(tokens):
                if tokens[i] in self.term_dict:
                    candidates = self.term_dict[tokens[i]]
                    for c in candidates:
                        # Checks whether normalized AllenNLP tokens equal the list
                        # of string tokens defining the term in the dictionary
                        if i + len(c) <= len(tokens):
                            equal = True
                            for j in range(len(c)):
                                if tokens[i + j] != c[j]:
                                    equal = False
                                    break

                            # If tokens match, labels the instance tokens
                            if equal:
                                for j in range(i + 1, i + len(c)):
                                    links[j] = 1
                                i = i + len(c) - 1
                                break
                i += 1

            return links

    lf = ExtractedPhrase(dict_full)
    lf.apply(docs)

    class ConsecutiveCapitals(LinkingRule):
        def apply_instance(self, instance):
            links = [0] * len(instance['tokens'])
            # We skip the first pair since the first
            # token is almost always capitalized
            for i in range(2, len(instance['tokens'])):
                # We skip this token if it all capitals
                all_caps = True
                text = instance['tokens'][i].text
                for char in text:
                    if char.islower():
                        all_caps = False
                        break

                if not all_caps and text[0].isupper(
                ) and instance['tokens'][i - 1].text[0].isupper():
                    links[i] = 1

            return links

    lf = ConsecutiveCapitals()
    lf.apply(docs)

    return docs


def ncbi_annotators(docs):
    dict_core = set()
    dict_core_exact = set()
    with open('../Dependency/AutoNER_dicts/NCBI/dict_core.txt') as f:
        for line in f.readlines():
            line = line.strip().split()
            term = tuple(line[1:])

            if len(term) > 1 or len(term[0]) > 3:
                dict_core.add(term)
            else:
                dict_core_exact.add(term)

    # Prepends common modifiers
    to_add = set()
    for term in dict_core:
        to_add.add(("inherited",) + term)
        to_add.add(("Inherited",) + term)
        to_add.add(("hereditary",) + term)
        to_add.add(("Hereditary",) + term)

    dict_core |= to_add

    # Removes common FP
    dict_core_exact.remove(("WT1",))
    dict_core_exact.remove(("VHL",))

    dict_full = set()

    with open('../Dependency/AutoNER_dicts/NCBI/dict_full.txt') as f:
        for line in f.readlines():
            line = line.strip().split()
            dict_full.add(tuple(line))

    lf = DictionaryMatcher(
        "CoreDictionaryUncased",
        dict_core,
        uncased=True,
        i_label="I")
    lf.apply(docs)

    lf = DictionaryMatcher("CoreDictionaryExact", dict_core_exact, i_label="I")
    lf.apply(docs)

    class CancerLike(TaggingRule):
        def apply_instance(self, instance):
            tokens = [token.text.lower() for token in instance['tokens']]
            labels = ['ABS'] * len(tokens)

            suffixes = ("edema", "toma", "coma", "noma")

            for i, token in enumerate(tokens):
                for suffix in suffixes:
                    if token.endswith(suffix) or token.endswith(suffix + "s"):
                        labels[i] = 'I'
            return labels

    lf = CancerLike()
    lf.apply(docs)

    class CommonSuffixes(TaggingRule):

        suffixes = {
            "agia",
            "cardia",
            "trophy",
            "toxic",
            "itis",
            "emia",
            "pathy",
            "plasia"}

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens'])):
                for suffix in self.suffixes:
                    if instance['tokens'][i].lemma_.endswith(suffix):
                        labels[i] = 'I'
            return labels

    lf = CommonSuffixes()
    lf.apply(docs)

    class Deficiency(TaggingRule):

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            # "___ deficiency"
            for i in range(len(instance['tokens']) - 1):
                if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i + 1].lemma_ == 'deficiency':
                    labels[i] = 'I'
                    labels[i + 1] = 'I'

                    # Adds any other compound tokens before the phrase
                    for j in range(i - 1, -1, -1):
                        if instance['tokens'][j].dep_ == 'compound':
                            labels[j] = 'I'
                        else:
                            break

            # "deficiency of ___"
            for i in range(len(instance['tokens']) - 2):
                if instance['tokens'][i].lemma_ == 'deficiency' and instance['tokens'][i + 1].lemma_ == 'of':
                    labels[i] = 'I'
                    labels[i + 1] = 'I'
                    nnp_active = False
                    for j in range(i + 2, len(instance['tokens'])):
                        if instance['tokens'][j].pos_ in ('NOUN', 'PROPN'):
                            if not nnp_active:
                                nnp_active = True
                        elif nnp_active:
                            break
                        labels[j] = 'I'

            return labels

    lf = Deficiency()
    lf.apply(docs)

    class Disorder(TaggingRule):

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens']) - 1):
                if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                                   1].lemma_ == 'disorder':
                    labels[i] = 'I'
                    labels[i + 1] = 'I'

                    # Adds any other compound tokens before the phrase
                    for j in range(i - 1, -1, -1):
                        if instance['tokens'][j].dep_ == 'compound':
                            labels[j] = 'I'
                        else:
                            break

            return labels

    lf = Disorder()
    lf.apply(docs)

    class Lesion(TaggingRule):

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens']) - 1):
                if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                                   1].lemma_ == 'lesion':
                    labels[i] = 'I'
                    labels[i + 1] = 'I'

                    # Adds any other compound tokens before the phrase
                    for j in range(i - 1, -1, -1):
                        if instance['tokens'][j].dep_ == 'compound':
                            labels[j] = 'I'
                        else:
                            break

            return labels

    lf = Lesion()
    lf.apply(docs)

    class Syndrome(TaggingRule):

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens']) - 1):
                if instance['tokens'][i].dep_ == 'compound' and instance['tokens'][i +
                                                                                   1].lemma_ == 'syndrome':
                    labels[i] = 'I'
                    labels[i + 1] = 'I'

                    # Adds any other compound tokens before the phrase
                    for j in range(i - 1, -1, -1):
                        if instance['tokens'][j].dep_ == 'compound':
                            labels[j] = 'I'
                        else:
                            break

            return labels

    lf = Syndrome()
    lf.apply(docs)

    terms = []
    with open('../Dependency/umls/umls_body_part.txt', 'r') as f:
        for line in f.readlines():
            terms.append(line.strip().split(" "))
    lf = DictionaryMatcher("TEMP", terms, i_label='TEMP', uncased=True, match_lemmas=True)
    lf.apply(docs)

    # noinspection PyShadowingNames
    class BodyTerms(TaggingRule):
        def apply_instance(self, instance):
            tokens = [token.text.lower() for token in instance['tokens']]
            labels = ['ABS'] * len(tokens)

            terms = {"cancer", "cancers", "damage", "disease", "diseases", "pain", "injury", "injuries"}

            for i in range(0, len(tokens) - 1):
                if instance['WISER_LABELS']['TEMP'][i] == 'TEMP':
                    if tokens[i + 1] in terms:
                        labels[i] = "I"
                        labels[i + 1] = "I"
            return labels

    lf = BodyTerms()
    lf.apply(docs)

    for doc in docs:
        del doc['WISER_LABELS']['TEMP']

    class OtherPOS(TaggingRule):
        other_pos = {"ADP", "ADV", "DET", "VERB"}

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(0, len(instance['tokens'])):
                if instance['tokens'][i].pos_ in self.other_pos:
                    labels[i] = "O"
            return labels

    lf = OtherPOS()
    lf.apply(docs)

    stop_words = {"a", "as", "be", "but", "do", "even",
                  "for", "from",
                  "had", "has", "have", "i", "in", "is", "its", "just",
                  "my", "no", "not", "on", "or",
                  "that", "the", "these", "this", "those", "to", "very",
                  "what", "which", "who", "with"}

    class StopWords(TaggingRule):

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens'])):
                if instance['tokens'][i].lemma_ in stop_words:
                    labels[i] = 'O'
            return labels

    lf = StopWords()
    lf.apply(docs)

    class Punctuation(TaggingRule):

        other_punc = {".", ",", "?", "!", ";", ":", "(", ")",
                      "%", "<", ">", "=", "+", "/", "\\"}

        def apply_instance(self, instance):
            labels = ['ABS'] * len(instance['tokens'])

            for i in range(len(instance['tokens'])):
                if instance['tokens'][i].text in self.other_punc:
                    labels[i] = 'O'
            return labels

    lf = Punctuation()
    lf.apply(docs)

    class PossessivePhrase(LinkingRule):
        def apply_instance(self, instance):
            links = [0] * len(instance['tokens'])
            for i in range(1, len(instance['tokens'])):
                if instance['tokens'][i - 1].text == "'s" or instance['tokens'][i].text == "'s":
                    links[i] = 1

            return links

    lf = PossessivePhrase()
    lf.apply(docs)

    class HyphenatedPhrase(LinkingRule):
        def apply_instance(self, instance):
            links = [0] * len(instance['tokens'])
            for i in range(1, len(instance['tokens'])):
                if instance['tokens'][i - 1].text == "-" or instance['tokens'][i].text == "-":
                    links[i] = 1

            return links

    lf = HyphenatedPhrase()
    lf.apply(docs)

    lf = ElmoLinkingRule(.8)
    lf.apply(docs)

    class CommonBigram(LinkingRule):
        def apply_instance(self, instance):
            links = [0] * len(instance['tokens'])
            tokens = [token.text.lower() for token in instance['tokens']]

            bigrams = {}
            for i in range(1, len(tokens)):
                bigram = tokens[i - 1], tokens[i]
                if bigram in bigrams:
                    bigrams[bigram] += 1
                else:
                    bigrams[bigram] = 1

            for i in range(1, len(tokens)):
                bigram = tokens[i - 1], tokens[i]
                count = bigrams[bigram]
                if count >= 6:
                    links[i] = 1

            return links

    lf = CommonBigram()
    lf.apply(docs)

    # noinspection PyShadowingNames
    class ExtractedPhrase(LinkingRule):
        def __init__(self, terms):
            self.term_dict = {}

            for term in terms:
                term = [token.lower() for token in term]
                if term[0] not in self.term_dict:
                    self.term_dict[term[0]] = []
                self.term_dict[term[0]].append(term)

            # Sorts the terms in decreasing order so that we match the longest
            # first
            for first_token in self.term_dict.keys():
                to_sort = self.term_dict[first_token]
                self.term_dict[first_token] = sorted(
                    to_sort, reverse=True, key=lambda x: len(x))

        def apply_instance(self, instance):
            tokens = [token.text.lower() for token in instance['tokens']]
            links = [0] * len(instance['tokens'])

            i = 0
            while i < len(tokens):
                if tokens[i] in self.term_dict:
                    candidates = self.term_dict[tokens[i]]
                    for c in candidates:
                        # Checks whether normalized AllenNLP tokens equal the list
                        # of string tokens defining the term in the dictionary
                        if i + len(c) <= len(tokens):
                            equal = True
                            for j in range(len(c)):
                                if tokens[i + j] != c[j]:
                                    equal = False
                                    break

                            # If tokens match, labels the instance tokens
                            if equal:
                                for j in range(i + 1, i + len(c)):
                                    links[j] = 1
                                i = i + len(c) - 1
                                break
                i += 1

            return links

    lf = ExtractedPhrase(dict_full)
    lf.apply(docs)

    return docs
