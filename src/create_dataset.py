# %%
# import json file into pandas dataframe
import pandas as pd
import numpy as np
import json
# %%
uid = []
content = []

# %%
def get_data(path):
    for line in open(path, "r").readlines():
        data = json.loads(line)
        if data["uid"] == "" or data["uid"].strip() == "NA":
            continue

        content_ = ""
        if data["title"] == "" or data["title"].strip() == "NA":
            pass
        else:
            content_ += data["title"] + "\n"
        
        if data["content"] == "" or data["content"].strip() == "NA":
            continue 

        uid.append(data["uid"])
        content.append(content_ + data["content"])

for path in ["../data/trn.json", "../data/tst.json"]:
    get_data(path)

# %%
# create a dataframe with column uid and column content
df = pd.DataFrame({"uid": uid, "content": content})

# get median length of strings in content
lengths = [len(c) for c in content]
print(np.median(lengths))
# get 95th percentile of lengths
print(np.percentile(lengths, 99.99))

# filter df so that we only keep rows, where "content" fields are within the 99th percentile of length
df = df[df["content"].apply(lambda x: len(x) < 21893.296199995093)]

# save dataframe to csv
df.to_csv("../data/df.csv", index=False)
