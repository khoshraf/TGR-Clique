# Temporal Graph Representation Learning via Maximal Cliques

## Introduction

This is the reference PyTorch implementation of the paper:
* Temporal Graph Representation Learning via Maximal Cliques


## Authors
Shima Khoshraftar, Aijun An and Nastaran Babanejad


## Requirements
* `python >= 3.7`, `PyTorch >= 1.4`, please refer to their official websites for installation details.
* Other dependencies:
```{bash}
pandas==0.24.2
tqdm==4.41.1
numpy==1.16.4
scikit_learn==0.22.1
matploblib==3.3.1
```

## Maximal clique generation
Maximal cliques of the training part of each dataset should be generated using the code provided in [here](https://github.com/darrenstrash/quick-cliques).

The output should be placed in the data directory. Two sample outputs for fb-messages dataset is at data/ml_fb_msg_train2.cliques and data/ml_fb_msg_train2_ind.cliques for transductive and inductive settings respectively.

## Running

#### Examples:

* To run **TGR-Clique** on fb-messages dataset in transductive training, sampling 10 length-2 walks for every node, with given maximal clique file:
```bash
python main.py -d fb_msg --mode t --clq_file data/ml_fb_msg_train2.cliques --num_walk 10 --len_walk 2 
```

* To run **TGR-Clique** on fb-messages dataset in inductive training, sampling 10 length-2 walks for every node, with the given maximal clique file:

```bash
python main.py -d fb_msg --mode i --clq_file data/ml_fb_msg_train2_ind.cliques --num_walk 10 --len_walk 2 
```



## Acknowledgement
Our implementation adapts the code [here](https://github.com/snap-stanford/CAW) as the code base and adapts it to our purpose. We thank the authors for sharing their code.

## Cite us
If you compare with, build on, or use aspects of the paper and/or code, please cite us:
```text
@inproceedings{
khoshraftar2022tgr,
title={Temporal Graph Representation Learning via Maximal Cliques},
author={Khoshraftar, Shima and An, Aijun and Babanejad, Nastaran},
booktitle={International Conference on Big Data 2022},
year={2022}
}
```

