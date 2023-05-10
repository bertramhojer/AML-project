# %%
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# %%
# load df
import pandas as pd
df = pd.read_csv("../data/df.csv")

# embed every sentence in the df["content"]
# %%
import time
# loop through df["content"] a 100 rows at a time
len_df = len(df)
embeddings = []
content = df["content"].to_numpy()
start = time.time()
step = 100
for i in range(0, len(df), step):
    emb = model.encode(content[i:min(i+step, len_df)], batch_size=64)
    embeddings.extend(model.encode(df["content"]))
    if i % 200 == 0:
        print(i)
print(time.time() - start)

# pickle embeddings to file
import pickle
with open("../data/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
# %%
