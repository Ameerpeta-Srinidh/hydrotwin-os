"""
HydroTwin OS — Plane 1: Thermal GNN Training

Generates synthetic data center layouts, numerically simulates heat diffusion
to get steady-state labels, and trains the ThermalGNN to approximate the solver.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse

from hydrotwin.physics.asset_graph import AssetGraph
from hydrotwin.physics.thermal_gnn import ThermalGNN, graph_to_tensors

def simulate_diffusion_numerically(graph: AssetGraph, it_load_kw: float = 10.0, cooling_kw: float = 5.0) -> torch.Tensor:
    """
    A simplified numerical relaxation solver for heat diffusion.
    We iterate to find steady-state temperatures to use as ground truth labels.
    """
    data = graph_to_tensors(list(graph.nodes.values()), list(graph.edges.values()))
    num_nodes = data["x"].shape[0]
    
    # Initialize temperatures at ambient 22C
    temps = torch.ones(num_nodes) * 22.0
    
    node_types = data["x"][:, 0:3] # One-hot: Rack, CRAH, PDU
    
    # Heat sources (+IT load) and sinks (-Cooling)
    heat_injection = (node_types[:, 0] * it_load_kw) - (node_types[:, 1] * cooling_kw)
    
    # Edges define thermal conductivity based on inverse distance
    adj = torch.zeros((num_nodes, num_nodes))
    edges = data["edge_index"]
    weights = data["edge_attr"]
    
    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        # Weight is 1 / distance, so higher weight = more conduction
        w = weights[i, 0] if weights.dim() > 1 else weights[i]
        adj[u, v] = w
        adj[v, u] = w
        
    row_sum = adj.sum(dim=1).unsqueeze(1)
    # Avoid div zero
    row_sum[row_sum == 0] = 1.0
    transition = adj / row_sum
    
    # Numerical relaxation: T_{t+1} = alpha * T_t + (1-alpha) * (Transition * T_t) + heat
    # Running for 50 steps to approximate steady state
    alpha = 0.5
    for _ in range(50):
        conduction = torch.matmul(transition, temps)
        temps = alpha * temps + (1 - alpha) * conduction + heat_injection * 0.1
        temps = torch.clamp(temps, 10.0, 60.0) # physical bounds
        
    return temps

def generate_dataset(num_samples: int):
    """Generate graphs and corresponding temperature labels."""
    dataset = []
    print(f"Generating {num_samples} synthetic layouts and simulating diffusion...")
    for _ in tqdm(range(num_samples)):
        num_racks = np.random.randint(4, 20)
        num_crahs = np.random.randint(1, 4)
        rows = np.random.randint(1, 5)
        
        graph = AssetGraph.create_synthetic(num_racks=num_racks, num_crahs=num_crahs, rows=rows)
        data = graph_to_tensors(list(graph.nodes.values()), list(graph.edges.values()))
        labels = simulate_diffusion_numerically(graph, 15.0, 200.0/num_crahs)
        
        dataset.append((data, labels))
    return dataset

def train_gnn(epochs: int = 50, num_samples: int = 1000):
    dataset = generate_dataset(num_samples)
    
    # Split train/test
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    test_data = dataset[split:]
    
    # Initialize GNN
    sample_data, _ = dataset[0]
    in_dim = sample_data["x"].shape[1]
    
    model = ThermalGNN(node_feature_dim=in_dim, hidden_dim=64, num_layers=4)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    print("Training Thermal GNN...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, labels in train_data:
            optimizer.zero_grad()
            out = model(data["x"], data["edge_index"], data["edge_attr"])
            
            # Predict only temperatures (1 output per node)
            loss = criterion(out["node_temps"], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss (MSE) = {total_loss/len(train_data):.4f}")
            
    # Evaluate
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, labels in test_data:
            out = model(data["x"], data["edge_index"], data["edge_attr"])
            test_loss += criterion(out["node_temps"], labels).item()
            
    mse = test_loss / len(test_data)
    rmse = np.sqrt(mse)
    
    print(f"\nFinal Test MSE: {mse:.4f} (Average error: {rmse:.2f}°C)")
    
    # Stress test on extreme topology
    print("Stress testing on extreme topology (100 Racks, 1 CRAH)...")
    extreme_graph = AssetGraph.create_synthetic(num_racks=100, num_crahs=1, rows=10)
    ext_data = graph_to_tensors(list(extreme_graph.nodes.values()), list(extreme_graph.edges.values()))
    with torch.no_grad():
        ext_out = model(ext_data["x"], ext_data["edge_index"], ext_data["edge_attr"])
        max_temp = ext_out["node_temps"].max().item()
        print(f"Extreme topology max predicted temp: {max_temp:.1f}°C")
        if max_temp > 100.0 or not np.isfinite(max_temp):
            print("WARNING: GNN gradients/weights exploded on OOD topology!")
        else:
            print("GNN remained stable under extreme topological stress.")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/thermal_gnn.pth")
    print("Model saved to models/thermal_gnn.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=2500)
    args = parser.parse_args()
    
    train_gnn(epochs=args.epochs, num_samples=args.samples)
