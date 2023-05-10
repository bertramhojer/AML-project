import pickle
import pandas as pd
import torch
import wandb
import numpy as np
import os
import glob
import random

from sklearn import metrics 

from tqdm.auto import tqdm
import torch.nn.functional as F

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from torch_geometric.loader import ClusterLoader, ClusterData, NeighborSampler
from torch_geometric.nn import SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric import data, transforms

log_steps = 1
num_workers = 12
num_layers = 2
hidden_channels = 256
dropout = 0.5
batch_size = 256
lr = 0.001
epochs = 250
eval_steps = 10
runs = 1

def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

dataset = PygNodePropPredDataset(name = 'ogbn-products') 
dataset = dataset[0]

total_len = dataset.x.size(0)

val_size, test_size = int(0.2 * total_len), int(0.2 * total_len)
seed_everything(42)
data = transforms.RandomNodeSplit(num_val=val_size, num_test=test_size)(dataset)
seed_everything(420)
subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                    batch_size=1024, shuffle=False,
                                    num_workers=num_workers)
seed_everything(69)
cluster_data = ClusterData(data, num_parts=15000,
                            recursive=False)
seed_everything(42069)
loader = ClusterLoader(cluster_data, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)



wandb.init(project="AML-exam", name=f'GCN_PCA_1')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def find_saved_model(path, pattern='gcn_pca*layer.pt'):
    saved_models = glob.glob(os.path.join(path, pattern))
    if saved_models:
        return max(saved_models, key=os.path.getctime)
    return None

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

    y_true_list = []
    y_pred_list = []

    for data in loader:
        data = data.to(device)
        if data.train_mask.sum() == 0:
            continue
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)[data.train_mask]
        y = data.y.squeeze(1)[data.train_mask]
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        # Calculate train loss, accuracy, and f1 score
        num_examples = data.train_mask.sum().item()
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        total_correct += out.argmax(dim=-1).eq(y).sum().item()

        y_true_list.append(y.cpu().detach().numpy())
        y_pred_list.append(out.argmax(dim=-1).cpu().detach().numpy())

    # Concatenate lists and compute f1 score
    y_true_agg = np.concatenate(y_true_list)
    y_pred_agg = np.concatenate(y_pred_list)
    f1_score_train = f1_score(y_true_agg, y_pred_agg, average='weighted')

    return total_loss / total_examples, total_correct / total_examples, f1_score_train

@torch.no_grad()
def test(model, data, subgraph_loader, device, test=False):
    model.eval()

    # Get predictions
    out = model.inference(data.x, subgraph_loader, device)
    out_softmax = F.log_softmax(out, dim=-1)
    y_true = data.y.squeeze(1)
    y_pred = out.argmax(dim=-1)

    # Accuracy
    train_acc =  accuracy_score(y_true[data.train_mask], y_pred[data.train_mask])
    valid_acc =  accuracy_score(y_true[data.val_mask], y_pred[data.val_mask])
    
    # F1_val
    f1_score_val = f1_score(y_true[data.val_mask], y_pred[data.val_mask], average='weighted')

    # Val loss
    val_loss = F.nll_loss(out_softmax[data.val_mask], y_true[data.val_mask]).item()
    
    if test:
        test_acc =  accuracy_score(y_true[data.test_mask], y_pred[data.test_mask])
        
        f1_score_test = f1_score(y_true[data.test_mask], y_pred[data.test_mask], average='weighted')
        return test_acc, f1_score_test
    
    return train_acc, valid_acc, f1_score_val, val_loss




model = SAGE(data.x.size(-1), hidden_channels, 47,
                num_layers, dropout).to(device)

for run in range(runs):
    saved_model_path = find_saved_model('../data/')

    if saved_model_path:
        print(f"Loading saved model from {saved_model_path}")
        checkpoint = torch.load(saved_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("No saved model found. Initializing a new model.")
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, 1 + epochs):
        loss, train_acc, f1_score_train = train(model, loader, optimizer, device)
        if epoch % log_steps == 0:
            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Approx Train Acc: {train_acc:.4f},'
                    f'F1 train: {f1_score_train:.4f}')
            wandb.log({"epoch": epoch, "train_loss": loss, "train_accuracy": train_acc, "train_f1": f1_score_train})

        if epoch >= 0 and epoch % eval_steps == 0:
            
            torch.save({    
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'../data/checkpoint/gcn_model_pca_{num_layers}layer_{epoch=}.pt')
            
            train_acc, valid_acc, f1_score_val, val_loss = test(model, data, subgraph_loader, device)

            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc}% '
                    f'F1 val: {100 * f1_score_val}%'
                    f'Val loss: {val_loss:.4f}')
            wandb.log({"epoch": epoch, "train_accuracy_full": train_acc, "val_accuracy": valid_acc, "val_f1": f1_score_val, "val_loss": val_loss})

torch.save({    
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'../data/gcn_model_pca_{num_layers}layer.pt')

test_acc, f1_score_test = test(model, data, subgraph_loader, device, test=True)
wandb.log({"test_acc": test_acc, "f1_score_test": f1_score_test, "num_parameters": count_parameters(model)})
wandb.finish()

