import numpy as np
import pandas
from sklearn import metrics


def evaluate(docs, target_sources, labels_to_map=None, labels_to_keep=None):
    """
    Extracts the evaluation results for one or more sources, and add them to a pandas DataFrame.
    """

    if isinstance(target_sources, str):
        target_sources = [target_sources]

    records = []
    for source in target_sources:
        results = get_results(docs, source, labels_to_map, labels_to_keep)
        # Put the results into a pandas dataframe
        for name in sorted(labels_to_keep) + ["micro", "weighted", "macro"]:
            if name in results:
                record = results[name]
                record["label"] = name
                record["model"] = source
                if name in labels_to_keep:
                    record["proportion"] = results["label_weights"][name]
                records.append(record)

    df = pandas.DataFrame.from_records(records)
    df["proportion"] = df.proportion.apply(lambda x: "%.1f %%" % (x * 100) if not np.isnan(x) else "")
    df["token_cee"] = df.token_cee.apply(lambda x: str(x) if not np.isnan(x) else "")
    df = df.set_index(["label", "proportion", "model"]).sort_index()
    df = df[["token_precision", "token_recall", "token_f1", "token_cee",
             "entity_precision", "entity_recall", "entity_f1"]]
    print("HERE")
    return df


def get_results(docs, target_source, labels_to_map=None, labels_to_keep=None, conf_threshold=0.5):
    """Computes the usual metrics (precision, recall, F1, cross-entropy) on the dataset, using the spacy entities
    in each document as gold standard, and the annotations of a given lb_source as the predicted values"""

    # We start by computing the TP, FP and FN values
    tok_tp = {}
    tok_fp = {}
    tok_fn = {}

    tok_logloss = 0
    tok_nb = 0

    ent_tp = {}
    ent_fp = {}
    ent_fn = {}
    ent_support = {}
    tok_support = {}

    for doc in docs:

        source_annotations = doc.user_data["annotations"][target_source]

        # We may need to do some mapping / filtering on the entities (eg. mapping PERSON to PER),
        # depending on the corpus we are dealing with
        spans = set()
        for (start, end), vals in source_annotations.items():
            if len(vals) > 0:
                best_val, best_conf = sorted(vals, key=lambda x: x[1])[-1]
                if labels_to_map is not None:
                    best_val = labels_to_map.get(best_val, best_val)
                if labels_to_keep is not None and best_val not in labels_to_keep:
                    continue
                elif best_conf >= conf_threshold:
                    spans.add((start, end, best_val))

        all_labels = {ent.label_ for ent in doc.ents} | {label for _, _, label in spans}
        for label in all_labels:
            true_spans = {(ent.start, ent.end) for ent in doc.ents if ent.label_ == label}
            pred_spans = {(start, end) for (start, end, label2) in spans if label2 == label}

            # Normalisation of dates (with or without prepositions / articles)
            if label == "DATE":
                true_spans = {
                    (start + 1 if doc[start].lower_ in {"in", "on", "a", "the", "for", "an", "at"} else start, end)
                    for (start, end) in true_spans}
                pred_spans = {
                    (start + 1 if doc[start].lower_ in {"in", "on", "a", "the", "for", "an", "at"} else start, end)
                    for (start, end) in pred_spans}

            ent_tp[label] = ent_tp.get(label, 0) + len(true_spans.intersection(pred_spans))
            ent_fp[label] = ent_fp.get(label, 0) + len(pred_spans - true_spans)
            ent_fn[label] = ent_fn.get(label, 0) + len(true_spans - pred_spans)
            ent_support[label] = ent_support.get(label, 0) + len(true_spans)

            true_tok_labels = {i for start, end in true_spans for i in range(start, end)}
            pred_tok_labels = {i for start, end in pred_spans for i in range(start, end)}
            tok_tp[label] = tok_tp.get(label, 0) + len(true_tok_labels.intersection(pred_tok_labels))
            tok_fp[label] = tok_fp.get(label, 0) + len(pred_tok_labels - true_tok_labels)
            tok_fn[label] = tok_fn.get(label, 0) + len(true_tok_labels - pred_tok_labels)
            tok_support[label] = tok_support.get(label, 0) + len(true_tok_labels)

        if len(doc.ents) > 0:
            tok_logloss += compute_logloss(doc, target_source, labels_to_map)
        tok_nb += len(doc)

    # We then compute the metrics themselves
    results = {}
    for label in ent_support:
        ent_pred = ent_tp[label] + ent_fp[label] + 1E-10
        ent_true = ent_tp[label] + ent_fn[label] + 1E-10
        tok_pred = tok_tp[label] + tok_fp[label] + 1E-10
        tok_true = tok_tp[label] + tok_fn[label] + 1E-10
        results[label] = {}
        results[label]["entity_precision"] = round(ent_tp[label] / ent_pred, 3)
        results[label]["entity_recall"] = round(ent_tp[label] / ent_true, 3)
        results[label]["token_precision"] = round(tok_tp[label] / tok_pred, 3)
        results[label]["token_recall"] = round(tok_tp[label] / tok_true, 3)

        ent_f1_numerator = (results[label]["entity_precision"] * results[label]["entity_recall"])
        ent_f1_denominator = (results[label]["entity_precision"] + results[label]["entity_recall"]) + 1E-10
        results[label]["entity_f1"] = 2 * round(ent_f1_numerator / ent_f1_denominator, 3)

        tok_f1_numerator = (results[label]["token_precision"] * results[label]["token_recall"])
        tok_f1_denominator = (results[label]["token_precision"] + results[label]["token_recall"]) + 1E-10
        results[label]["token_f1"] = 2 * round(tok_f1_numerator / tok_f1_denominator, 3)

    results["macro"] = {"entity_precision": np.round(np.mean([results[lb]["entity_precision"] for lb in ent_support]),
                                                     3),
                        "entity_recall": np.round(np.mean([results[lb]["entity_recall"] for lb in ent_support]), 3),
                        "token_precision": np.round(np.mean([results[lb]["token_precision"] for lb in ent_support]), 3),
                        "token_recall": np.round(np.mean([results[lb]["token_recall"] for lb in ent_support]), 3)}

    label_weights = {lb: ent_support[lb] / sum(ent_support.values()) for lb in ent_support}
    results["label_weights"] = label_weights
    results["weighted"] = {"entity_precision": np.round(np.sum([results[lb]["entity_precision"] * label_weights[lb]
                                                                for lb in ent_support]), 3),
                           "entity_recall": np.round(np.sum([results[lb]["entity_recall"] * label_weights[lb]
                                                             for lb in ent_support]), 3),
                           "token_precision": np.round(np.sum([results[lb]["token_precision"] * label_weights[lb]
                                                               for lb in ent_support]), 3),
                           "token_recall": np.round(np.sum([results[lb]["token_recall"] * label_weights[lb]
                                                            for lb in ent_support]), 3)}

    ent_pred = sum([ent_tp[lb] for lb in ent_support]) + sum([ent_fp[lb] for lb in ent_support]) + 1E-10
    ent_true = sum([ent_tp[lb] for lb in ent_support]) + sum([ent_fn[lb] for lb in ent_support]) + 1E-10
    tok_pred = sum([tok_tp[lb] for lb in ent_support]) + sum([tok_fp[lb] for lb in ent_support]) + 1E-10
    tok_true = sum([tok_tp[lb] for lb in ent_support]) + sum([tok_fn[lb] for lb in ent_support]) + 1E-10
    results["micro"] = {"entity_precision": round(sum([ent_tp[lb] for lb in ent_support]) / ent_pred, 3),
                        "entity_recall": round(sum([ent_tp[lb] for lb in ent_support]) / ent_true, 3),
                        "token_precision": round(sum([tok_tp[lb] for lb in ent_support]) / tok_pred, 3),
                        "token_recall": round(sum([tok_tp[lb] for lb in ent_support]) / tok_true, 3),
                        "token_cee": round(tok_logloss / tok_nb, 3)}

    for metric in ["macro", "weighted", "micro"]:
        ent_f1_numerator = (results[metric]["entity_precision"] * results[metric]["entity_recall"])
        ent_f1_denominator = (results[metric]["entity_precision"] + results[metric]["entity_recall"]) + 1E-10
        results[metric]["entity_f1"] = 2 * round(ent_f1_numerator / ent_f1_denominator, 3)

        tok_f1_numerator = (results[metric]["token_precision"] * results[metric]["token_recall"])
        tok_f1_denominator = (results[metric]["token_precision"] + results[metric]["token_recall"]) + 1E-10
        results[metric]["token_f1"] = 2 * round(tok_f1_numerator / tok_f1_denominator, 3)

    return results


def compute_logloss(doc, target_source, labels_to_map=None):
    all_labels = {ent.label_ for ent in doc.ents}
    pos_labels = ["O"] + ["%s-%s" % (bilu, label) for label in sorted(all_labels) for bilu in "BILU"]
    pos_label_indices = {pos_label: i for i, pos_label in enumerate(pos_labels)}

    gold_probs = np.zeros((len(doc), 1 + len(all_labels) * 4))
    for ent in doc.ents:
        if ent.end == ent.start + 1:
            index = pos_label_indices["U-%s" % ent.label_]
            gold_probs[ent.start, index] = 1
        else:
            index = pos_label_indices["B-%s" % ent.label_]
            gold_probs[ent.start, index] = 1
            for i in range(ent.start + 1, ent.end - 1):
                index = pos_label_indices["I-%s" % ent.label_]
                gold_probs[i, index] = 1
            index = pos_label_indices["L-%s" % ent.label_]
            gold_probs[ent.end - 1, index] = 1
    gold_probs[:, 0] = 1 - gold_probs.sum(axis=1)

    pred_probs = np.zeros(gold_probs.shape)
    for (start, end), vals in doc.user_data["annotations"][target_source].items():
        if end > len(doc):
            print("bad boundary")
            end = len(doc)
        for label, conf in vals:
            if labels_to_map is not None:
                label = labels_to_map.get(label, label)
            if label not in all_labels:
                continue
            if end == start + 1:
                index = pos_label_indices["U-%s" % label]
                pred_probs[start, index] = conf
            else:
                index = pos_label_indices["B-%s" % label]
                pred_probs[start, index] = conf
                for i in range(start + 1, end - 1):
                    index = pos_label_indices["I-%s" % label]
                    pred_probs[i, index] = conf
                index = pos_label_indices["L-%s" % label]
                pred_probs[end - 1, index] = conf

    pred_probs[:, 0] = 1 - pred_probs.sum(axis=1)
    loss = metrics.log_loss(gold_probs, pred_probs, normalize=False)
    return loss
