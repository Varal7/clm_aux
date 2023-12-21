import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, BatchNorm
from torch_geometric.data import Data, Dataset

def get_node_features(sentence):
    return torch.ones(128)

class CustomNLIDataset(Dataset):
    def __init__(self, scores, labels, lengths, sentences, transform=None, pre_transform=None):
        super(CustomNLIDataset, self).__init__(transform, pre_transform)
        self.scores = scores
        self.labels = labels
        self.lengths = lengths
        self.sentences = sentences

    def len(self):
        return len(self.labels)

    def get(self, idx):
        # Get the scores, labels, lengths, and sentences for the current index
        current_scores = self.scores[idx]
        current_labels = self.labels[idx]
        current_lengths = self.lengths[idx]
        current_sentences = self.sentences[idx]

        # Create node features (x) from current_sentences
        x = torch.tensor([get_node_features(sentence).tolist() for report in current_sentences for sentence in report], dtype=torch.float)

        # Create edge_index and edge_attr from current_scores
        edge_index = torch.tensor([(i, j) for i in range(current_scores.shape[0]) for j in range(current_scores.shape[1]) if i != j], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([current_scores[i, j].tolist() for i in range(current_scores.shape[0]) for j in range(current_scores.shape[1]) if i != j], dtype=torch.float)

        # Create the Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=current_labels, lengths=current_lengths)

        return data


class EdgeModel(nn.Module):
    def __init__(self, num_edge_features, num_node_features):
        super(EdgeModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_edge_features, num_node_features),
            nn.Tanh(),
            nn.Linear(num_node_features, num_node_features * num_node_features)
        )

    def forward(self, x):
        return self.layer(x)


class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes, dropout=0.5, num_layers=3):
        super(GNNModel, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.edge_model = EdgeModel(num_edge_features, num_node_features)

        self.layers = nn.Sequential(*[
                nn.Sequential(
                    NNConv(num_node_features, num_node_features, self.edge_model, aggr='mean'),
                    BatchNorm(num_node_features),
                    nn.Tanh(),
                    nn.Dropout(self.dropout),
                )
                for _ in range(self.num_layers)
            ]
        )

        self.fc1 = torch.nn.Linear(num_node_features * 4, num_node_features)
        self.fc2 = torch.nn.Linear(num_node_features, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, lengths = data.x, data.edge_index, data.edge_attr, data.lengths

        for layers in self.layers:
            x = layers[0](x, edge_index, edge_attr)
            for layer in layers[1:]:
                x = layer(x)

        hidden = x

        # TODO maybe use max, min, mean, sum too
        indices = lengths.cumsum(dim=0)
        indices = torch.cat([torch.tensor([0], device=indices.device), indices])
        zeros = torch.zeros(1, hidden.size(1), device=hidden.device)

        parts = [hidden[indices[i]:indices[i+1]] if indices[i] != indices[i+1] else zeros for i in range(len(indices)-1)]

        mean_pool = torch.stack([torch.mean(part, dim=0) for part in parts])
        max_pool = torch.stack([torch.max(part, dim=0)[0] for part in parts])
        min_pool = torch.stack([torch.min(part, dim=0)[0] for part in parts])
        sum_pool = torch.stack([torch.sum(part, dim=0) for part in parts])

        pooled = torch.cat([mean_pool, max_pool, min_pool, sum_pool], dim=1)

        # Fully connected layer
        fc = self.fc1(pooled)
        fc = torch.tanh(fc)
        fc = F.dropout(fc, p=self.dropout, training=self.training)
        fc = self.fc2(fc)

        out = F.log_softmax(fc, dim=1)

        return hidden, pooled, fc, out
