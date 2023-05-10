from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
import pandas as pd
import pickle
import numpy as np

with open("../data/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# load df
embeddings_list = pickle.load(open("../data/embeddings-list.pkl", "rb"))
df = pd.read_csv("data/df.csv")
df["embeddings"] = embeddings_list
df_content = df["content"]
df = df.drop(columns=["content"])

# load idx2asin
idx2asin = pd.read_csv("../data/nodeidx2asin.csv")
#rename first column in idx2asin to uid
idx2asin = idx2asin.rename(columns={"asin": "uid"})

#change "node idx" to type int64
idx2asin["node idx"] = idx2asin["node idx"].astype("int64")

dataset = PygNodePropPredDataset(name = "ogbn-products") 
idx2asin["y"] = dataset.y

# merge df and idx2asin on the column "uid". I want the join that keeps all the rows :)
df2 = idx2asin.merge(df, on="uid", how="left")

#remove all rows that contain NaN in embeddings column
df2 = df2.dropna(subset=["embeddings"])

# add the index of df as a column
df2["index"] = list(range(len(df2)))

# create a dictionary that maps node idx to index
node_ids = dict(zip(df2["node idx"], df2["index"]))

# in dataset.edge_list only keep colums that contain a value present in node_ids.values()
values = node_ids.keys()

#remove any columns that contain a value not present in values
edge_index = dataset.edge_index
edge_index = edge_index[:, np.isin(edge_index[0], list(values)) & np.isin(edge_index[1], list(values))]

# convert every entry in edge_index by applying 
edge_index = np.vectorize(node_ids.get)(edge_index)

x = np.array(df2['embeddings'].tolist())

data_ = Data(x=x, edge_index=edge_index, y=df2.y.to_numpy())

with open("dataset.pkl", "wb") as o:
    data_ = pickle.dump(data_, o)