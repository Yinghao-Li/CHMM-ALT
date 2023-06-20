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
Notice that only the packages used for model training are listed in the file.
Those for dataset construction are not listed for the reason mentioned below.

## 2. Dataset Construction

### Pre-Processed Datasets

The dataset construction program depends on several external libraries such as `AllenNLP`, `wiser` or `skweak`, some of which have conflict dependencies, some are no longer maintained.
Building datasets from the source data could be hard under this situation.
Hence, we directly post the pre-processed datasets in `.json` format under the `data` directory for reproduction.
If you prefer building the dataset from source, you can refer to the following subsection.

### Source Data

The dataset construction program for the `NCBI-Disease`, `BC5CDR` and `LaptopReview` datasets is modified from the `wiser` project ([paper](http://cs.brown.edu/people/sbach/files/safranchik-aaai20.pdf))
that contains three repos.

The dataset construction program for the `CoNLL 2003` dataset is based on [skweak](https://github.com/NorskRegnesentral/skweak).


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

We have three entry files: `chmm.py`, `bert.py` and `alt.py`, which are all stored in the `run`.
Each file corresponds to a component in our alternate training pipeline.
In the `scripts` folder are the configuration files that defines the hyperparameters for model training.
You can either use `.json` or `.sh`.
Please make sure you are at the project directory (`[]/CHMM-ALT/`).

To train and evaluate CHMM, go to `./label_model/` and run
```shell
PYHTONPATH="." CUDA_VISIBLE_DEVICES=0 python ./run/chmm.py ./scripts/config_chmm.json
```
Here `conig_chmm.json` is a configuration file only for demonstration.
Another option is
```shell
sh ./scripts/run_chmm.sh
```
You need to fine-tune the hyper-parameters to get better performance.

The way to run BERT-NER (`./run/bert.py`) or Alternation training (`./run/alt.py`) is similar and will not be detailed.

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

