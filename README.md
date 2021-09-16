# CHMM-ALT

Alternate-training of Conditional hidden Markov model and BERT-NER.

This code accompanies the paper [BERTifying the Hidden Markov Model for Multi-Source Weakly Supervised Named Entity Recognition](https://arxiv.org/abs/2105.12848).

> To view the previous version of program used for the paper, switch to branch `prev`.

## 1. Dependency
Please check `requirement.txt` for the package dependency requirement.
The data construction program may need the specified versions of `spaCy` and `AllenNLP`.
The model training program should be compatible with any package versions.


## 2. Dataset Construction

The dataset construction program for the `NCBI-Disease`, `BC5CDR` and `LaptopReview` datasets is modified from the wiser project ([paper](http://cs.brown.edu/people/sbach/files/safranchik-aaai20.pdf))
that contains three repos.

The dataset construction program for the `CoNLL 2003` dataset is based on [skweak](https://github.com/NorskRegnesentral/skweak).


### Source Data

The source data are provided in the folders `DataConstr/<DATASET NAME>/data`.
You can also download the source data from the links below:

* [BC5CDR](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/): Download the train, development, and test BioCreative V CDR corpus data files.

* [NCBI Disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/): Download the complete training, development, and testing sets.

* [LaptopReview](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools): Download the train data V2.0 for the Laptops and Restaurants dataset and the test data - phase B.

* [CoNLL 2003](https://www.clips.uantwerpen.be/conll2003/ner/): You can find a pre-processed CoNLL 2003 English dataset [here](https://github.com/ningshixian/NER-CONLL2003/tree/master/data).

Place the downloaded data in the corresponding folders `DataConstr/<DATASET NAME>/data`.

### External Dependencies

To build CoNLL 2003 dataset, you may need to get the external dictionaries and models on which `skweak` depends.

You can get these files from [here](https://github.com/NorskRegnesentral/skweak/releases).
Unzip them and place the outputs into `DataConstr/Dependency/` for usage.

### Building datasets

Run the `build.sh` script in the dataset folder `DataConstr/<DATASET NAME>` with 
```
./build.sh
```
You will see `train.json`, `valid.json`, `test.json` and `meta.json` files in your target folder if the program runs successfully.

You can also customize the script with your favorite arguments.

### Backward compatibility
**Notice:** the datasets contructed in the way above are not completely the same as the datasets used in the paper.

However, our code has fully support to the previous version of datasets.
To reproduce the results in the paper, please refer to the dataset construction methods in the `prev` branch and link the file location arguments to their directories.


## 3. Run

We use the argument parsing techniques from the Huggingface `transformers` [repo](https://github.com/huggingface/transformers) in our program.
It supports the orginary argument parsing approach from shell inputs as well as parsing from `json` files.


### Conditional hidden Markov model

To train and evaluate CHMM, go to `./LabelModel/` and run
```shell
python chmm_train.py config.json
```
Here `conig.json` is just a demo configuration.
You need to fine-tune the hyper-parameters to get better performance.

### BERT-NER

You can train a fully-supervised BERT-NER model with ground truth labels by going to the `./EndModel/` folder and run
```shell
python bert_train.py config.json
```

### Alternate training

The file `./ALT/chmm-alt.py` realizes the alternate training technique introduced in the paper.
you can train a CHMM and a BERT alternately with
```shell
./chmm-alt.sh
```
or
```
python chmm-alt.py config.json
```


## 4. Citation

If you find our work helpful, you can cite it as 

```
@inproceedings{li-etal-2021-bertifying,
    title = "{BERT}ifying the Hidden {M}arkov Model for Multi-Source Weakly Supervised Named Entity Recognition",
    author = "Li, Yinghao  and
      Shetty, Pranav  and
      Liu, Lucas  and
      Zhang, Chao  and
      Song, Le",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.482",
    doi = "10.18653/v1/2021.acl-long.482",
    pages = "6178--6190",
}
```

