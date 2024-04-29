'''On fera ici tourner la méthode GAE'''
import numpy as np
import os
import csv
from extract_graphs import read_txt, read_nodes, create_training_graph
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import torch_geometric.nn as graphnn
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn.conv  import GATConv
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''Import the Graph, work on its nodes'''
G = create_training_graph()
nodes = G.nodes(data = True)
node_list = [node for node, attr in nodes]
labels_nodes_list = [attr['label'] for node, attr in nodes]

x = torch.tensor(node_list, dtype=torch.long)
y = torch.tensor(labels_nodes_list, dtype=torch.float)

'''Work on the edges'''
true_edges = [(u,v) for u,v,d in G.edges(data = True) if d['exist'] == 1]
false_edges = [(u,v) for u,v,d in G.edges(data = True) if d['exist'] == 0]
true_edges_torch = torch.tensor(true_edges, dtype=torch.long)
true_edges_torch = true_edges_torch.t().contiguous()    #Torch array of true edges

all_edges_torch = torch.tensor([(u,v) for u,v,d in G.edges(data = True)])   #all edges
labels_edges_torch = torch.tensor([d['exist'] for u,v,d in G.edges(data = True)])   #labels of the edges (true or false: 1 or 0)


'''Define a mapping where all nodes id are contiguous (needed to use torch_geometric)'''

torch_node_ids = torch.tensor(node_list)
node_id_mapping = {node_id.item(): i for i, node_id in enumerate(torch_node_ids)} #Definition of the mapping
node_inverse_mapping = {i: node_id for i, node_id in enumerate(torch_node_ids)} #inverse the mapping

'''Use the mapping tu update the link names'''
#for all links
edge_list_updated = torch.tensor([(node_id_mapping[u.item()],node_id_mapping[v.item()]) for u,v in all_edges_torch])  

#for true links
true_edges_torch_updated = true_edges_torch.clone()
for old_id, new_id in node_id_mapping.items():
    true_edges_torch_updated[true_edges_torch == old_id] = new_id
total_graph = Data(edge_index=true_edges_torch_updated,x=x,y=y).to(device)

'''Define the dataset of edges using the mapping for edges'''

class EdgeDataSet(torch.utils.data.Dataset):
    def __init__(self, torch_edge_list, labels):
        self.torch_edge_list = torch_edge_list
        self.labels_edges = labels

    def __len__(self):
        return len(self.torch_edge_list)

    def __getitem__(self, idx):
        edge = self.torch_edge_list[idx]
        edge = torch.tensor([node_id_mapping[edge[0].item()], node_id_mapping[edge[1].item()],idx]) #(node_0, node_1, index of the link in the edge list)
        label = self.labels_edges[idx]
        return edge, label
    
        
dataset_tot = EdgeDataSet(all_edges_torch, labels_edges_torch)

'''Create the training and testing datasets'''
total_length = len(dataset_tot)
train_length = int(total_length * 0.9)  # 90% for training
test_length = total_length - train_length  # 10% for testing
train_dataset, test_dataset = random_split(dataset_tot, [train_length, test_length])

'''find the true edges in the training part (with the new mapping)'''
train_indices = train_dataset.indices
true_train_indices = []
for index in train_indices:
    if labels_edges_torch[index] == 1:
        true_train_indices.append(index)
train_edges_true = edge_list_updated[true_train_indices]    
train_edges_true = train_edges_true.t().contiguous().to(device)



'''Create the dataloaders'''
def collate_gpu(batch):
    x, t = torch.utils.data.default_collate(batch)
    return x.to(device= device), t.to(device= device)

train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, collate_fn=collate_gpu)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_gpu)

'''Define network model'''

class BasicGraphModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.graphconv1 = graphnn.GCNConv(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)  # Dropout layer
        self.graphconv2 = graphnn.GCNConv(hidden_size, hidden_size)
        self.elu = nn.ELU()
        self.graphconv3 = graphnn.GCNConv(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.graphconv1(x, edge_index)
        x = self.dropout(x)
        x = self.elu(x)
        x = self.graphconv2(x, edge_index)
        x = self.dropout(x)
        x = self.elu(x)
        x = self.graphconv3(x, edge_index)
        return x
    
'''Prepare the training of the model'''

num_epochs = 100
input_size = 932
hidden_size = 32
output_size = 16


GNN = BasicGraphModel(input_size, hidden_size, output_size).to(device)
loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(GNN.parameters(), lr=0.01)#,weight_decay=0.001)

'''First prediction before any training (to compare with the other predictions)'''

GNN.eval()
li_z = GNN(total_graph.y, train_edges_true)
for (edges, labels_batch) in test_dataloader:
    labels_batch = labels_batch.to(device)
    li_scal_prod = torch.sum(li_z[edges[:,0],:] * li_z[edges[:,1],:], dim=1)
    li_predic = (li_scal_prod>0).cpu().detach()
    li_true = labels_batch.cpu().detach()
print('first evaluation accuracy:{}'.format(accuracy_score(li_true.cpu().numpy(),li_predic.cpu().numpy())))

'''Train the model'''
li_accuracy = []
li_accu_train = []
for epoch in range(num_epochs):
    GNN.train()
    for i, (edges, labels_batch) in enumerate(train_dataloader):
        li_z = GNN(total_graph.y, train_edges_true)
        li_scal_prod = torch.sum(li_z[edges[:,0],:] * li_z[edges[:,1],:], dim=1)
        labels_batch = labels_batch.to(device)
        loss_val = loss(li_scal_prod, labels_batch.float())
        optimizer.zero_grad()
        loss_val.backward()
        grad_norms = [p.grad.norm().item() for p in GNN.parameters() if p.grad is not None] #make sure the grad norm is not 0 (that's why I calculate it here)
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        optimizer.step()
        if i%10 ==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}],\
                train Loss: {loss_val.item()},train accuracy{accuracy_score(labels_batch.cpu().detach().numpy(),(li_scal_prod>0).cpu().detach().numpy())}','grad norm:',avg_grad_norm)
        accuracy_train = accuracy_score(labels_batch.cpu().detach().numpy(),(li_scal_prod>0).cpu().detach().numpy())
        li_accu_train.append(accuracy_train)
    
    #evaluate the model after each epoch
    GNN.eval()
    li_z = GNN(total_graph.y, train_edges_true)
    for (edges, labels_batch) in test_dataloader:
        labels_batch = labels_batch.to(device)
        li_scal_prod = torch.sum(li_z[edges[:,0],:] * li_z[edges[:,1],:], dim=1)
        li_predic = (li_scal_prod>0).cpu().detach()
        li_true = labels_batch.cpu().detach()
    accuracy = accuracy_score(li_true.cpu().numpy(),li_predic.cpu().numpy())
    li_accuracy.append(accuracy)
    print('test set accuracy:{}'.format(accuracy_score(li_true.cpu().numpy(),li_predic.cpu().numpy())))

'''Plot the accuracy'''
plt.plot(li_accuracy, label = 'accuracy testing set')
plt.plot(li_accu_train, label = 'accuracy training set')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

    