# CHMM-ALT

**Al**ternate-**t**raining for **C**onditional **h**idden **M**arkov **m**odel and BERT-NER.

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Yinghao-Li/CHMM-ALT)
![GitHub stars](https://img.shields.io/github/stars/Yinghao-Li/CHMM-ALT.svg?color=gold)
![GitHub forks](https://img.shields.io/github/forks/Yinghao-Li/CHMM-ALT?color=9cf)


This code accompanies the paper [BERTifying the Hidden Markov Model for Multi-Source Weakly Supervised Named Entity Recognition](https://arxiv.org/abs/2105.12848).

> To view the previous version of the program used for the paper, switch to branch `prev`.

Conditional hidden Markov model (CHMM) is also included in the [Wrench project 🔧](https://github.com/JieyuZ2/wrench)

Check out my follow-up to this work: [Sparse-CHMM](https://github.com/Yinghao-Li/Sparse-CHMM)

## 1. Dependencies
Please check `requirement.txt` for the package dependency requirement.
The data construction program may need the specified versions of `spaCy` and `AllenNLP`.
The model training program should be compatible with any package version.

## 2. Dataset Construction

The dataset construction program for the `NCBI-Disease`, `BC5CDR` and `LaptopReview` datasets is modified from the `wiser` project ([paper](http://cs.brown.edu/people/sbach/files/safranchik-aaai20.pdf))
that contains three repos.

The dataset construction program for the `CoNLL 2003` dataset is based on [skweak](https://github.com/NorskRegnesentral/skweak).


### Source Data

The source data are provided in the folders `data_constr/<DATASET NAME>/data`.
You can also download the source data from the links below:

* [BC5CDR](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/): Download the train, development, and test BioCreative V CDR corpus data files.

* [NCBI Disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/): Download the complete training, development, and testing sets.

* [LaptopReview](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools): Download the train data V2.0 for the Laptops and Restaurants dataset and the test data - phase B.

* [CoNLL 2003](https://www.clips.uantwerpen.be/conll2003/ner/): You can find a pre-processed CoNLL 2003 English dataset [here](https://github.com/ningshixian/NER-CONLL2003/tree/master/data).

Place the downloaded data in the corresponding folders `data_constr/<DATASET NAME>/data`.

### External Dependencies

To build datasets, you may need to get the external dictionaries and models on which `skweak` and `wiser` depends.

You can get the files from [Google Drive](https://drive.google.com/file/d/1BaSQ2rQvAA8ecgIc3KDtmUpGvdvDzr5S/view?usp=sharing) or download them individually from [here](https://github.com/NorskRegnesentral/skweak/releases) and [here](https://github.com/BatsResearch/safranchik-aaai20-code).
Unzip them and place the outputs into `data_constr/Dependency/` for usage.

### Building datasets

Run the `build.sh` script in the dataset folder `data_constr/<DATASET NAME>` with 
```
./build.sh
```
You will see `train.json`, `valid.json`, `test.json`, and `meta.json` files in your target folder if the program runs successfully.

You can also customize the script with your favorite arguments.

### Backward compatibility
**Notice:** the datasets constructed in the way above are not completely the same as the datasets used in the paper.

However, our code has full support for the previous version of datasets.
To reproduce the results in the paper, please refer to the dataset construction methods in the `prev` branch and link the file location arguments to their directories.

**Note:** Our data format is <span style="color:red">not compatible</span> with [Wrench](https://github.com/JieyuZ2/wrench/issues/9).

## 3. Run

We use the argument parsing techniques from the Huggingface `transformers` [repo](https://github.com/huggingface/transformers) in our program.
It supports the ordinary argument parsing approach from shell inputs as well as parsing from `json` files.

To try the code, clone this repo or your forked repo into the local machine and follow the instructions below.
Notice that this repo contains a submodule, which will not be automatically downloaded with `clone`.
To fetch the submodule content, use `git submodule update --init`.
When you update your local repo with `git pull`, be sure to run `git submodule update --remote` to get the submodule updates.

### Conditional hidden Markov model

To train and evaluate CHMM, go to `./label_model/` and run
```shell
python chmm_train.py config.json
```
Here `conig.json` is just a demo configuration.
You need to fine-tune the hyper-parameters to get better performance.

### BERT-NER

You can train a fully-supervised BERT-NER model with ground truth labels by going to the `./end_model/` folder and run
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

