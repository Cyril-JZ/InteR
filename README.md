# Synergistic Interplay between Search and Large Language Models for Information Retrieval

This repository is the official implementation of ACL 2024 paper: [*Synergistic Interplay between Search and Large Language Models for Information Retrieval*](https://arxiv.org/abs/2305.07402).

## 1. create environment
```shell
conda create -n inter python=3.8
conda activate inter
pip install -r requirements.txt
pip install -U openai pyserini
```

## 2. download index and passage data 
```shell
mkdir ./indexes/
wget https://www.dropbox.com/s/rf24cgsqetwbykr/lucene-index-msmarco-passage.tgz?dl=0
wget https://www.dropbox.com/s/5vhl1aynl0kg3rj/contriever_msmarco_index.tar.gz?dl=0
# then unzip two tgz files into ./indexes/

mkdir ./data_msmarco/
wget https://www.dropbox.com/s/yms13b9k850b3vt/collection.tsv?dl=0
# then put tsv file into ./data_msmarco/
```

## 3. run InteR
You may edit the **openai key** in `main.py` first.

```shell
mkdir ./runs_inter
./run_dl19.sh  # for DL'19
./run_dl20.sh  # for DL'20
```


### If you find this work helpful, please cite our paper:
```latex
@misc{feng2023synergistic,
      title={Synergistic Interplay between Search and Large Language Models for Information Retrieval}, 
      author={Jiazhan Feng and Chongyang Tao and Xiubo Geng and Tao Shen and Can Xu and Guodong Long and Dongyan Zhao and Daxin Jiang},
      year={2023},
      eprint={2305.07402},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
