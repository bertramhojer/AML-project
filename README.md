# Cluster GCN w. MLP Models

The repository was developed by:
Bjarke Christensen, Bertram Højer, Karl Kieler & Oliver Jarvis

This project is a variation of the 'ogbn-products' classification task [https://ogb.stanford.edu/docs/nodeprop/#ogbn-products]. We attempt to solve a node-level classification task. The original dataset is a graph of amazon products purchased together where node-level features are PCA derived representations of a bag-of-words embedding of product descriptions. We have updated the graph such that node-level features are instead BERT-embeddings and compare a simple MLP model to a ClusterGCN model. 

## Data
In order to train the embeddings models, the dataset is needed. The dataset can downloaded from here.


https://drive.google.com/file/d/1UckqCj6lwNViA3LVEwl6Y_i7KpPAfqpd/view?fbclid=IwAR2k90VTIVtT2NCUfovO0lNTZrsI1dG8kUOfEdgQaBIFT5WeWF9UVeKcPyw


Once downloaded, add the .csv to the data folder.

## Preprocessing
Instead of downloading the data, you can also pre-process it yourself.
This requires downloading the following files, unpacking them and adding them to the data folder.


Amazon-3M: https://drive.google.com/file/d/1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN/view?usp=sharing


nodeidx2asin.csv: https://drive.google.com/file/d/1hxWJ3e_Jfk9HCdOqKSEk3HMFqpylkgak/view?usp=share_link

## Embeddings
Word embeddings are created using the 'SentenceTransformers' library for all original product descriptions. We used the MiniLM model [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2]. All matching nodes in the graph have then been updated with the new embedding features. When updating the graph we go from 2.45M to 2.35M nodes.

## Models
### Multi-Layer Perceptron
The MLP models were trained in google colab notebooks:

- Bag-of-words implementation: https://colab.research.google.com/drive/1CafDsj3n39SBXOpTZZtThuFbu5J8ZU4N?usp=sharing

- Sbert implementation: https://colab.research.google.com/drive/1vu1sAFRL2sGWkHGQ0EWl1td0KNR6gdDd?usp=sharing


The MLP we train has a very simple architecture. We simply wanted to assess potential differences in using the graph structure of the problem for classification or simply using the embeddings. 

| Embedding | Parameters | Acc.     | F1       |
|-----------|------------|----------|----------|
| BOW       | 221,999    | 73.6 %   | 72,2 %   |
| BERT      | 367,457    | 86.6 %   | 86.2 %   |

### ClusterGCN
General GCNs have a bottleneck in terms of efficiency and scalability. ClusterGCN attempts to combat this problem by partitioning the graph into subgraphs and performing convolutions on these to reduce time and memory complexity. The output of each cluster is aggregated to obtain the final prediction. The file for training is ogbn.py and ogbn_pca.py

| Embedding | Parameters | Acc.     | F1       |
|-----------|------------|----------|----------|
| BOW       | 358,128    | 87.5 %   | 86.5 %   |
| BERT      | 358,128    | 90.1 %   | 90.6 %   |

## Repo structure 
```bash 
.
├── README.md
├── models
│   ├── bow-model-checkpoint-100.pt
│   ├── embedding-model-checkpoint-150.pt
│   ├── gcn_model_embedding_2layer_epoch500.pt
│   └── gcn_model_pca_2layer_epoch250.pt
├── requirements.txt
└── src
    ├── create_dataset.py
    ├── create_embeddings.py
    ├── create_graph_dataset.py
    ├── evaluation.ipynb
    ├── ogbn.py
    └── ogbn_pca.py
```