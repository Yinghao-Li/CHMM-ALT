# WISER

Welcome to WISER (*Weak and Indirect Supervision for Entity Recognition*), a system for training sequence tagging models, particularly neural networks for named entity recognition (NER) and related tasks. WISER uses *weak supervision* in the form of rules to train these models, as opposed to hand-labeled training data.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

The original WISER paper can be accessed [here](http://cs.brown.edu/people/sbach/files/safranchik-aaai20.pdf).

## Benchmarks

| Method | NCBI-Disease (F1) | BC5CDR (F1) | LaptopReview (F1) |
| ------------- |-------------| -----| -----|
| AutoNER | 75.52 | 82.13 | 65.44 |
| Snorkel | 73.41 | 82.24 | 63.54 |
| WISER | **79.03** | **82.94** | **69.04** |

## Getting Started

These instructions will WISER up and running on your local machine to develop your own pipelines for weakly supervised for sequence tagging tasks.

### Installing

WISER requires Python 3.7. To install the required dependencies, please run

```
pip install -r requirements.txt
```

Or alternatively

```
conda install --file requirements.txt
```

Then, inside the *wiser* directory, please run

```
pip install .
```

## Getting Started

Refer to *tutorial/introduction* for a comprehensive introduction to using WISER to train end-to-end frameworks with weak supervision. More tutorials coming soon!

Once you're comfortable with the WISER framework, we recommend looking at our [FAQ](https://github.com/BatsResearch/wiser/blob/master/FAQ.md) for strategies on how to write rules and debug your pipeline.

## Citation

Please cite the following paper if you are using our tool. Thank you!

[Esteban Safranchik](https://www.linkedin.com/in/safranchik/), Shiying Luo, [Stephen H. Bach](http://cs.brown.edu/people/sbach/). "Weakly Supervised Sequence Tagging From Noisy Rules". In 34th AAAI Conference on Artificial Intelligence, 2020.

```
@inproceedings{safranchik2020weakly,
  title = {Weakly Supervised Sequence Tagging From Noisy Rules}, 
  author = {Safranchik, Esteban and Luo, Shiying and Bach, Stephen H.}, 
  booktitle = {AAAI}, 
  year = 2020, 
}
```

# Weakly Supervised Sequence Tagging from Noisy Rules - Reproducibility Code

## Getting Started

These instructions will get you a copy of our experiments up and running on your local machine for development and testing purposes.

### Installing

In your virtual environment, please install the required dependencies using

```
pip install -r requirements.txt
```
Or alternatively
```
conda install --file requirements.txt
```

## Datasets

Our experiments depend on *six* different datasets that you will need to download separately.

* [BC5CDR](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/): Download and install the train, development, and test BioCreative V CDR corpus data files. Place the three separate files inside data/BC5CD

* [NCBI Disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/): Download and install the complete training, development, and testing sets. Place the three separate files inside *data/NCBI*.

* [LaptopReview](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools): Download the train data V2.0 for the Laptops and Restaurants dataset, and place the *Laptop_Train_v2.xml* file inside *data/LaptopReview*. Then, download the test data - phase B, and place the *Laptops_Test_Data_phaseB.xml* file inside the same directory.

* [CoNLL v5](https://catalog.ldc.upenn.edu/LDC2013T19): Download and compile the English dataset version 5.0, and place it in *data/conll-formatted-ontonotes-5.0*.

* [Scibert](https://github.com/allenai/scibert): Download the scibert-scivocab-uncased version of the Scibert embeddings, and place the files *weights.tar.gz and *vocab.txt* inside *data/scibert_scibocab_uncased*.

* [UMLS](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html): The UMLS dictionaries have been extracted from the UMLS 2018AB dataset and are provided in our code. They are distributed according to the [License Agreement for Use of the UMLS® Metathesaurus®](https://uts.nlm.nih.gov/help/license/LicenseAgreement.pdf).

* [AutoNER Dictionaries](https://github.com/shangjingbo1226/AutoNER). The AutoNER dictionaries for the BC5CDR, LaptopReview, and NCBI datasets have been  generously provided by Jingbo Shang et al. They have been sourced from the EMNLP 2018 paper "Learning Named Entity Tagger using Domain-Specific Dictionary".

## Citation

Please cite the following paper if you are using our tool. Thank you!

Safranchik Esteban, Shiying Luo, Stephen H. Bach. "Weakly Supervised Sequence Tagging From Noisy Rules". In 34th AAAI Conference on Artificial Intelligence, 2020.

```
@inproceedings{safranchik2020weakly,
  title = {Weakly Supervised Sequence Tagging From Noisy Rules}, 
  author = {Safranchik, Esteban and Luo, Shiying and Bach, Stephen H.}, 
  booktitle = {AAAI}, 
  year = 2020, 
}
```

# Label Models

[![Build Status](https://travis-ci.com/BatsResearch/labelmodels.svg?token=sinAgJjnTsxQ2oN3R9vi&branch=master)](https://travis-ci.com/BatsResearch/labelmodels)

Lightweight implementations of generative label models for weakly supervised machine learning

# Example Usage
```python
# Let votes be an m x n matrix where m is the number of data examples, n is the
# number of label sources, and each element is in the set {0, 1, ..., k}, where
# k is the number of classes. If votes_{ij} is 0, it means that label source j
# abstains from voting on example i.

# As an example, we create a random votes matrix for binary classification with
# 1000 examples and 5 label sources
import numpy as np
votes = np.random.randint(0, 3, size=(1000, 5))

# We now can create a Naive Bayes generative model to estimate the accuracies
# of these label sources
from labelmodels import NaiveBayes

# We initialize the model by specifying that there are 2 classes (binary
# classification) and 5 label sources
model = NaiveBayes(num_classes=2, num_lfs=5)

# Next, we estimate the model's parameters
model.estimate_label_model(votes)
print(model.get_accuracies())

# We can obtain a posterior distribution over the true labels
labels = model.get_label_distribution(votes)
```
