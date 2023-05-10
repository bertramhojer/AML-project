import pickle
import pandas as pd
import torch
import wandb

from sklearn import metrics 

from tqdm.auto import tqdm
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import ClusterLoader, ClusterData, NeighborSampler
from torch_geometric.nn import SAGEConv

from ogb.nodeproppred import Evaluator
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric import data, transforms

wandb.init(project="")

# Load the PKL file
with open('../data/dataset.pkl', 'rb') as f:
    pkl_data = pickle.load(f)

pkl_data.x = torch.tensor(pkl_data.x, dtype=torch.float)
pkl_data.edge_index = torch.tensor(pkl_data.edge_index, dtype=torch.long)
pkl_data.y = torch.tensor(pkl_data.y, dtype=torch.long)

total_len = pkl_data.x.size(0)

val_size, test_size = int(0.1 * total_len), int(0.2 * total_len)
data = transforms.RandomNodeSplit(num_val=val_size, num_test=test_size)(pkl_data)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

log_steps = 1
num_workers = 12
num_layers = 1
hidden_channels = 256
dropout = 0.5
batch_size = 64
lr = 0.01
epochs = 50
eval_steps = 10
runs = 1

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return torch.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

def train(model, loader, optimizer, device):
    model.train()

    total_loss = total_examples = 0
    total_correct = total_examples = 0
    for data in loader:
        data = data.to(device)
        if data.train_mask.sum() == 0:
            continue
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)[data.train_mask]
        y = data.y[data.train_mask]
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        num_examples = data.train_mask.sum().item()
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        total_correct += out.argmax(dim=-1).eq(y).sum().item()

    return total_loss / total_examples, total_correct / total_examples

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

@torch.no_grad()
def test(model, data, subgraph_loader, device):
    model.eval()

    out = model.inference(data.x, subgraph_loader, device)

    y_true = data.y
    y_pred = out.argmax(dim=-1)

    train_acc =  accuracy_score(y_true[data.train_mask], y_pred[data.train_mask])
    valid_acc =  accuracy_score(y_true[data.val_mask], y_pred[data.val_mask])
   
    f1_score_val = f1_score(y_true[data.val_mask], y_pred[data.val_mask], average='weighted')
     
    
    return train_acc, valid_acc, f1_score_val

subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                    batch_size=1024, shuffle=False,
                                    num_workers=num_workers)

model = SAGE(data.x.size(-1), hidden_channels, 47,
                num_layers, dropout).to(device)

cluster_data = ClusterData(data, num_parts=15000,
                            recursive=False, save_dir="../data/cluster/")

loader = ClusterLoader(cluster_data, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)

# get me the number of model params in my pytorch model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


sum(p.numel() for p in model.parameters() if p.requires_grad)

for run in range(runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, 1 + epochs):
        loss, train_acc = train(model, loader, optimizer, device)
        if epoch % log_steps == 0:
            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Approx Train Acc: {train_acc:.4f}')
            wandb.log({"epoch": epoch, "train_loss": loss, "train_acc": train_acc})

        if epoch >= 10 and epoch % eval_steps == 0:
            train_acc, valid_acc, f1_score_val = test(model, data, subgraph_loader, device)
            #valid_acc, f1_score_val = result
            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc}% '
                    f'F1 val: {100 * f1_score_val}%')
            wandb.log({"epoch": epoch, "train_acc": train_acc, "valid_acc": valid_acc, "f1_score_val": f1_score_val})

wandb.finish()