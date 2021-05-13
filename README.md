# AlT-NER

Alternate-training for multi-source weakly supervised Named Entity Recognition

## 1. Dependency
- python 3.6 [link](https://www.python.org/)
- pytorch 1.6.0 [link](https://pytorch.org/)
- transformers 3.4.0 [link](https://github.com/huggingface/transformers)
- pytokenizations [link](https://github.com/tamuhey/tokenizations)
- NLTK [link](https://www.nltk.org/)

Note: this project is only tested under the given environment.
Other library versions may cause unexpected behaviors.

## 2. Dataset Construction

### Constructing NCBI-Disease, BC5CDR and LaptopReview Dataset

The dataset construction code is modified from the wiser project ([paper](http://cs.brown.edu/people/sbach/files/safranchik-aaai20.pdf))
that contains three repos. We combined these repos and removed unecessary files.
The original repos are [repo1](https://github.com/BatsResearch/wiser), [repo2](https://github.com/BatsResearch/labelmodels) and [repo3](https://github.com/BatsResearch/safranchik-aaai2020-code).

#### Requirement
Note that the requirement to run the dataset construction project differs from the requirement to run our own project.
We suggest to create a new virtual environment within conda to avoid conflict.

- allennlp 1.1.0
- spacy 2.3.2
- pytokenizations
- NLTK
- pytorch


#### Source Data (Modified from the original projects)

You can download the source data and external dictionaries from the websites below:

* [BC5CDR](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/): Download and install the train, development, and test BioCreative V CDR corpus data files. Place the three separate files inside `DatasetConstruction/wiser/data/BC5CDR`.

* [NCBI Disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/): Download and install the complete training, development, and testing sets. Place the three separate files inside `DatasetConstruction/wiser/data/NCBI`.

* [LaptopReview](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools): Download the train data V2.0 for the Laptops and Restaurants dataset, and place the *Laptop_Train_v2.xml* file inside `DatasetConstruction/wiser/data/LaptopReview`. Then, download the test data - phase B, and place the *Laptops_Test_Data_phaseB.xml* file inside the same directory.

* [Scibert](https://github.com/allenai/scibert): Download the scibert-scivocab-uncased version of the Scibert embeddings, and place the files *weights.tar.gz and *vocab.txt* inside `DatasetConstruction/wiser/data/scibert_scibocab_uncased`.

* [UMLS](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html): The UMLS dictionaries have been extracted from the UMLS 2018AB dataset and are provided in our code. They are distributed according to the [License Agreement for Use of the UMLS® Metathesaurus®](https://uts.nlm.nih.gov/help/license/LicenseAgreement.pdf).

* [AutoNER Dictionaries](https://github.com/shangjingbo1226/AutoNER). The AutoNER dictionaries for the BC5CDR, LaptopReview, and NCBI datasets have been  generously provided by Jingbo Shang et al. They have been sourced from the EMNLP 2018 paper "Learning Named Entity Tagger using Domain-Specific Dictionary".

#### Construct datasets

Go to the folders `DatasetConstruction/wiser/BC5CDR`, `DatasetConstruction/wiser/NCBI` and `DatasetConstruction/wiser/LaptopReview`,
use the jupyter notebook file (`*.ipynb`) to construct the datasets, and move the generated `*.pt` files into the corresponding folders in `./data/`.

### Constructing CoNLL 2003 Dataset

The dataset construction code is modified from [this work](http://cs.brown.edu/people/sbach/files/safranchik-aaai20.pdf).


#### Requirement
Note that the requirement to run the dataset construction project differs from the requirement to run our own project.
We suggest to create a new virtual environment within conda to avoid conflict.

- spacy 2.2.2
- pytokenizations
- NLTK
- pytorch
- hmmlearn
- snips-nlu-parsers
- numba

In addition, you need to download additional dictionary files. The instructions are listed in the original repo.
The dictionry files should be put into `DatasetConstruction/Weak-NER/data` folder.

#### Construct datasets

Go to the folders `DatasetConstruction/Weak-NER`, you need to run `Build-SpaCy-docs.ipynb` and `Data-Constr-Pipeline.ipynb` in sequence to get the `*.pt` files.
After that, move the generated `*.pt` files into `./data/Co03`.


## 3. Run

Go to the root directory of this project then use the scripts included to start the training and evaluation process.
For example:
```shell script
./run_laptop.sh 5
```
The argument is the GPU id.
Currently this project only partially support parallel training.
We do not recommend running the project on multiple GPUs as this function has not been tested.

