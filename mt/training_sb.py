import gc
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.nn import Linear, ReLU, Sequential
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn import (
    AttentionalAggregation,
    GATv2Conv,
    GCNConv,
    GINEConv,
    global_mean_pool,
)
from torch_geometric.nn.aggr import AttentionalAggregation
from tqdm import tqdm
import itertools

from mt.definitions import REPO_DIR

g_types = {"unk": 0, "ast": 1, "ch22": 2, "cfg": 3, "dg": 4, "cfg_ast": 5}
repo_dir = REPO_DIR / "pytorch/vision"
result_dir = repo_dir / "results"
result_dir.mkdir(exist_ok=True)
SAMPL_NODES, SAMPL_ITER, SAMPL_NEIGHBORS = 750_000, 1, 10_000


class NodeFeatureEmbedding(nn.Module):
    def __init__(self, num_x, num_y, emb_x, emb_y):
        super(NodeFeatureEmbedding, self).__init__()
        self.embedding_x = nn.Embedding(num_x, emb_x)
        self.embedding_y = nn.Embedding(num_y, emb_y)

    def forward(self, x):
        # Assuming x is of shape (num_nodes, 2), where each entry is a category index
        x_embedding = self.embedding_x(x[:, 0])
        y_embedding = self.embedding_y(x[:, 1])
        # Concatenate the embeddings along the last dimension
        return torch.cat([x_embedding, y_embedding], dim=-1)


class EdgeFeatureEmbedding(nn.Module):
    def __init__(self, num_x, num_y, emb_x, emb_y):
        super(EdgeFeatureEmbedding, self).__init__()
        self.embedding_x = nn.Embedding(num_x, emb_x)
        self.embedding_y = nn.Embedding(num_y, emb_y)

    def forward(self, edge_attr):
        # Assuming edge_attr is of shape (num_edges, 2), where each entry is a category index
        x_embedding = self.embedding_x(edge_attr[:, 0])
        y_embedding = self.embedding_y(edge_attr[:, 1])
        # Concatenate the embeddings along the last dimension
        return torch.cat([x_embedding, y_embedding], dim=-1)


# Simple
class GCNModel(nn.Module):
    def __init__(
        self, node_embedding_dim, edge_embedding_dim, hidden_channels, out_channels=1
    ):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(node_embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_embedding_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(
                hidden_channels, 1
            ),  # Transform edge attributes to scalar weights
        )

        self.regressor = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_weight = self.edge_mlp(
            edge_attr
        ).squeeze()  # Transform edge attributes to edge weights

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))

        x = global_mean_pool(x, batch)

        return self.regressor(x).squeeze()


# Attention based
class GATModel(nn.Module):
    def __init__(
        self,
        node_embedding_dim,
        edge_embedding_dim,
        hidden_channels,
        agg_hidden_channels,
        out_channels=1,
        num_heads=2,
    ):
        super(GATModel, self).__init__()
        self.conv1 = GATv2Conv(
            node_embedding_dim,
            hidden_channels,
            heads=num_heads,
            concat=True,
            edge_dim=edge_embedding_dim,
        )
        self.conv2 = GATv2Conv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            concat=True,
            edge_dim=edge_embedding_dim,
        )
        self.conv3 = GATv2Conv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=1,
            concat=False,
            edge_dim=edge_embedding_dim,
        )

        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, agg_hidden_channels),
            nn.ReLU(),
            nn.Linear(agg_hidden_channels, 1),
        )

        self.attention_aggregation = AttentionalAggregation(gate_nn=self.gate_nn)

        self.regressor = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)

        x = self.attention_aggregation(x, batch)

        return self.regressor(x).squeeze()


# Structure, sequence based
class GINEModel(nn.Module):
    def __init__(
        self,
        node_embedding_dim,
        edge_embedding_dim,
        hidden_channels,
        lstm_hidden_dim,
        out_channels=1,
    ):
        super(GINEModel, self).__init__()
        nn1 = Sequential(
            Linear(node_embedding_dim, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        nn2 = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        nn3 = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
        )

        self.conv1 = GINEConv(nn1, edge_dim=edge_embedding_dim)
        self.conv2 = GINEConv(nn2, edge_dim=edge_embedding_dim)
        self.conv3 = GINEConv(nn3, edge_dim=edge_embedding_dim)

        self.lstm = nn.LSTM(hidden_channels, lstm_hidden_dim, batch_first=True)

        self.regressor = nn.Linear(lstm_hidden_dim, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        # Aggregate node embeddings into a graph-level embedding
        x = global_mean_pool(x, batch)

        # Expand dimensions to fit LSTM input requirements: (batch_size, seq_len, input_dim)
        x = x.unsqueeze(1)  # Assuming each graph is treated as a single sequence

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Use the output of the last LSTM cell
        x = x[:, -1, :]

        return self.regressor(x).squeeze()


class GraphDataset(Dataset):
    def __init__(self, repo_dir: Path, graph_type: str) -> None:
        self.graph_type = graph_type
        self.repo_dir = repo_dir
        with open(repo_dir / "maps.json") as f:
            self.maps = json.load(f)

        self.node_type = len(self.maps[graph_type]["nodes"])
        self.g_type_node = len(g_types) - 1
        self.g_type_edge = len(g_types) - 1 if graph_type != "all" else len(g_types)
        self.edge_type = len(self.maps[graph_type]["edges"])

        self.root_dir = repo_dir / "pts"
        with open(self.repo_dir / "residuals.pkl", "rb") as f:
            self.targets = pickle.load(f)
        self.files = list(self.root_dir.glob(f"{self.graph_type}_*.pt"))
        self.num_samples = len(self.files)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Data:
        commit_sha, graph = torch.load(self.root_dir / f"{self.graph_type}_{idx}.pt")
        target = self.targets[commit_sha]
        graph.y = torch.tensor(target, dtype=torch.float32)  # Add target to graph data
        return graph


def train_model(
    node_emb_model,
    edge_emb_model,
    model,
    train_dataset,
    val_dataset,
    optimizer,
    criterion,
    device,
    num_epochs=50,
    patience=10,
):
    model.to(device)
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for graph in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{num_epochs}"):
            train_loader = NeighborLoader(
                graph,
                num_neighbors=[SAMPL_NEIGHBORS] * SAMPL_ITER,
                batch_size=SAMPL_NODES,
                input_nodes=torch.arange(graph.num_nodes),
                shuffle=True,
            )
            for batch in tqdm(train_loader, leave=False):
                batch = batch.to(device)
                optimizer.zero_grad()
                emb_x = node_emb_model(batch.x)
                edge_index = batch.edge_index
                emb_edge_attr = edge_emb_model(batch.edge_attr)
                output = model(emb_x, edge_index, emb_edge_attr, batch.batch)
                targets = batch.y
                if targets.dim() == 0:  # Check if scalar tensor
                    targets = targets.unsqueeze(0)
                targets = targets[: batch.batch_size]
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for graph in val_dataset:
                val_loader = NeighborLoader(
                    graph,
                    num_neighbors=[SAMPL_NEIGHBORS] * SAMPL_ITER,
                    batch_size=SAMPL_NODES,
                    input_nodes=torch.arange(graph.num_nodes),
                    shuffle=False,
                )
                for batch in tqdm(val_loader, leave=False):
                    batch = batch.to(device)
                    emb_x = node_emb_model(batch.x)
                    edge_index = batch.edge_index
                    emb_edge_attr = edge_emb_model(batch.edge_attr)
                    output = model(emb_x, edge_index, emb_edge_attr, batch.batch)
                    targets = batch.y
                    if targets.dim() == 0:  # Check if scalar tensor
                        targets = targets.unsqueeze(0)
                    targets = targets[: batch.batch_size]
                    val_loss += criterion(output, targets).item()

        avg_val_loss = val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "node_emb": node_emb_model.state_dict(),
                    "edge_emb": edge_emb_model.state_dict(),
                },
                result_dir
                / f"{dataset.graph_type}_{model.__class__.__name__}_best_model.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        scheduler.step(avg_val_loss)

    state = torch.load(
        result_dir / f"{dataset.graph_type}_{model.__class__.__name__}_best_model.pt"
    )
    model.load_state_dict(state["model"])
    node_emb_model.load_state_dict(state["node_emb"])
    edge_emb_model.load_state_dict(state["edge_emb"])
    return model, train_losses, val_losses


def test_model(node_emb_model, edge_emb_model, model, test_dataset):
    model.eval()
    true_vals = []
    pred_vals = []
    with torch.no_grad():
        for graph in test_dataset:
            test_loader = NeighborLoader(
                graph,
                num_neighbors=[SAMPL_NEIGHBORS] * SAMPL_ITER,
                batch_size=SAMPL_NODES,
                input_nodes=torch.arange(graph.num_nodes),
                shuffle=False,
            )
            graph_true_vals = []
            graph_pred_vals = []
            for batch in tqdm(test_loader, leave=False):
                batch = batch.to(device)
                emb_x = node_emb_model(batch.x)
                edge_index = batch.edge_index
                emb_edge_attr = edge_emb_model(batch.edge_attr)
                output = model(emb_x, edge_index, emb_edge_attr, batch.batch)
                targets = batch.y
                if targets.dim() == 0:  # Check if scalar tensor
                    targets = targets.unsqueeze(0)
                targets = targets[: batch.batch_size]
                graph_true_vals.append(targets.cpu().numpy())
                graph_pred_vals.append(output.cpu().numpy())

            # Aggregate predictions for each graph
            true_vals.append(np.mean(graph_true_vals))
            pred_vals.append(np.mean(graph_pred_vals))

    return true_vals, pred_vals


graph_types = ["dg", "ast", "ch22", "cfg", "all"]
models = ["GCNModel", "GATModel", "GINEModel"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for graph_type in graph_types:
    dataset = GraphDataset(repo_dir, graph_type)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    for model_name in models:
        if (result_dir / f"{model_name}_{graph_type}.pt").exists():
            print(f"Skipping {model_name} on {graph_type} dataset...")
            continue
        node_emb_model = NodeFeatureEmbedding(
            dataset.node_type, dataset.g_type_node, 60, 4
        ).to(device)
        edge_emb_model = EdgeFeatureEmbedding(
            dataset.edge_type, dataset.g_type_edge, 60, 4
        ).to(device)
        if model_name == "GCNModel":
            model = GCNModel(64, 64, 128, 1).to(device)
        elif model_name == "GATModel":
            model = GATModel(64, 64, 64, 64, 1, 1).to(device)
        else:
            model = GINEModel(64, 64, 64, 64, 1).to(device)

        params = itertools.chain(
            model.parameters(), node_emb_model.parameters(), edge_emb_model.parameters()
        )
        optimizer = optim.Adam(
            params,
            lr=0.001,
        )
        criterion = nn.MSELoss()

        print(f"Training {model_name} on {graph_type} dataset...")
        model, train_losses, val_losses = train_model(
            node_emb_model,
            edge_emb_model,
            model,
            train_dataset,
            val_dataset,
            optimizer,
            criterion,
            device,
            num_epochs=50,
        )

        train_true, train_pred = test_model(
            node_emb_model, edge_emb_model, model, train_dataset
        )
        val_true, val_pred = test_model(
            node_emb_model, edge_emb_model, model, val_dataset
        )
        test_true, test_pred = test_model(
            node_emb_model, edge_emb_model, model, test_dataset
        )
        mse = mean_squared_error(test_true, test_pred)

        results = {
            "model": model_name,
            "graph_type": graph_type,
            "test_true": test_true,
            "test_pred": test_pred,
            "train_true": train_true,
            "train_pred": train_pred,
            "val_true": val_true,
            "val_pred": val_pred,
            "mse": mse,
        }

        with open(result_dir / f"{model_name}_{graph_type}.pkl", "wb") as f:
            pickle.dump(results, f)

        torch.save(model.state_dict(), result_dir / f"{model_name}_{graph_type}.pt")

        del model
        del optimizer
        del node_emb_model
        del edge_emb_model
        torch.cuda.empty_cache()
        gc.collect()

    del dataset
    del train_dataset
    del val_dataset
    del test_dataset
    torch.cuda.empty_cache()
    gc.collect()
