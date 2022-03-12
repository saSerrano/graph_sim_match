# Graph Similarity and Matching

This repository contains the implementation of two articles whose objective is to perform node matching across graphs and compute their similarity.

## Installation
Execute ```pip install -e .```. It is recommended to create a python environment (such as conda) with python 3.7.11.

## Graph Matching based on Linear Programming
File ```wgmp_lp.py``` contains the implementation of the graph-matching method, called ```wgmp_lp```, presented in the article [A Linear Programming Approach for the Weighted Graph Matching Problem](https://ieeexplore.ieee.org/abstract/document/211474?casa_token=RWt4qoAVaLkAAAAA:ZSdqi3GAb3gBiXE6uwpmOTs1zA06cqREXEsnrSXkxRUeEWrZt9DXu33yTLN67-y3bD2zylJnFTKN3To). The file also contains the ```paper_exp``` method, which can be used to replicate the evaluation of the graph-matching method, as in the article.

## Graph Similarity and Matching
File ```graph_sim.py``` contains the implementation of the node-edge-based similarity and matching algorithm proposed in the article [Graph similarity scoring and matching](https://www.sciencedirect.com/science/article/pii/S0893965907001012). The method called ```graph_match``` requires a pair of adjacency matrices and the list of names for the nodes in the pair of graphs to compute the node and edge similarity scores, as well as the node and edge matching. Results are saved in three files: two ```.csv``` files containing the node and edge similarity scores, respectively, and third file containing the node and edge matching.

## Examples
In the directory ```examples``` there are scripts that show how one can use both similarity methods.
