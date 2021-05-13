import spacy
import json
import numpy as np
from spacy import displacy, tokens
from tqdm.auto import tqdm
from Core.Util import tokenise_fast, Trie


def docbin_reader(docbin_input_file, vocab=None, cutoff=None, nb_to_skip=0):
    """Generate Spacy documents from the input file"""

    if vocab is None:
        if not hasattr(docbin_reader, "vocab"):
            print("Loading vocabulary", end="...", flush=True)
            docbin_reader.vocab = spacy.load("en_core_web_md").vocab
            print("done")
        vocab = docbin_reader.vocab
    vocab.strings.add("subtok")

    fd = open(docbin_input_file, "rb")
    data = fd.read()
    fd.close()
    docbin = spacy.tokens.DocBin(store_user_data=True)
    docbin.from_bytes(data)
    del data
    # print("Total number of documents in docbin:", _length(docbin))

    # Hack to easily skip a number of documents
    if nb_to_skip:
        docbin.tokens = docbin.tokens[nb_to_skip:]
        docbin.spaces = docbin.spaces[nb_to_skip:]
        docbin.user_data = docbin.user_data[nb_to_skip:]

    reader = docbin.get_docs(vocab)
    for i, doc in enumerate(reader):
        yield doc
        if cutoff is not None and (i + 1) >= cutoff:
            return


def extract_json_data(json_file):
    """Extract entities from a Json file """

    print("Extracting data from", json_file)
    fd = open(json_file)
    data = json.load(fd)
    fd.close()
    trie = Trie()
    for neClass, names in data.items():
        print("Populating trie for entity class %s (number: %i)" % (neClass, len(names)))
        for name in tqdm(names):

            # Removes parentheses and appositions
            name = name.split("(")[0].split(",")[0].strip()

            name = tuple(tokenise_fast(name))
            # Add the entity into the trie (values are tuples since each entity may have several possible types)
            if name in trie and neClass not in trie[name]:
                trie[name] = (*trie[name], neClass)
            else:
                trie[name] = (neClass,)
    return trie


def display_entities(spacy_doc, source=None):
    """Display the entities in a spacy document (with some preprocessing to handle special characters)"""

    if source is None:
        spans = [(ent.start, ent.end, ent.label_) for ent in spacy_doc.ents]
    else:
        spans = [(start, end, label) for start, end in spacy_doc.user_data["annotations"][source]
                 for label, conf in spacy_doc.user_data["annotations"][source][(start, end)] if conf > 0.2]

    text = spacy_doc.text
    # Due to a rendering bug, we need to escape dollar signs, and keep the offsets to get the
    # entity boundaries right in respect to the text
    dollar_sign_offsets = [i for i in range(len(text)) if text[i] == "$"]
    text = text.replace("$", r"\$")

    entities = {}
    for start, end, label in spans:

        start_char = spacy_doc[start].idx
        end_char = spacy_doc[end - 1].idx + len(spacy_doc[end - 1])

        # We need to pad the character offsets for escaped dollar signs
        if dollar_sign_offsets:
            start_char += np.searchsorted(dollar_sign_offsets, start_char)
            end_char += np.searchsorted(dollar_sign_offsets, end_char)

        if (start_char, end_char) not in entities:
            entities[(start_char, end_char)] = label

        # If we have several alternative labels for a span, join them with +
        elif label not in entities[(start_char, end_char)]:
            entities[(start_char, end_char)] = entities[(start_char, end_char)] + "+" + label

    entities = [{"start": start, "end": end, "label": label} for (start, end), label in entities.items()]
    doc2 = {"text": text, "title": None, "ents": entities}
    spacy.displacy.render(doc2, jupyter=True, style="ent", manual=True)
