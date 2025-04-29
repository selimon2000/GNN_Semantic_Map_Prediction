import sys
sys.path.append('/home/selimon/capstone_v3/GNN')
from data_set import *

import random
import copy
import matplotlib.pyplot as plt

import torch
from torch.utils.data import random_split
import networkx as nx
from torch_geometric.loader import DataLoader



# DEGREE BASED NODE REMOVAL
# REMOVES NODES WITH LOWER DEGREES
# BETTER THAN ABOVE FUNCTION AS IT MEANS THAT THERE ARE LESS DISCONNECTED NODES (IN MY CASE BEING PORTALS) 
def remove_room_nodes(data, remove_ratio=0.5):
    
    # Get device from input data
    device = data['room'].x.device
    
    # Calculate degrees of room nodes
    degrees = torch.zeros(data['room'].num_nodes, dtype=torch.long, device=device)
    
    if ('room', 'connects', 'portal') in data.edge_index_dict:
        src_nodes = data['room', 'connects', 'portal'].edge_index[0]
        degrees.scatter_add_(0, src_nodes, torch.ones(src_nodes.size(0), dtype=torch.long, device=device))
    
    if ('portal', 'connects', 'room') in data.edge_index_dict:
        dst_nodes = data['portal', 'connects', 'room'].edge_index[1]
        degrees.scatter_add_(0, dst_nodes, torch.ones(dst_nodes.size(0), dtype=torch.long, device=device))
    
    # Sort nodes by degree (ascending) and get indices
    sorted_indices = torch.argsort(degrees)
    num_to_remove = int(remove_ratio * data['room'].num_nodes)
    remove_indices = set(sorted_indices[:num_to_remove].tolist())
    keep_indices = sorted(set(range(data['room'].num_nodes)) - remove_indices)
    
    # Create mapping from old to new indices
    old_to_new = {old: new for new, old in enumerate(keep_indices)}
    
    # Filter room node features and labels
    data['room'].x = data['room'].x[keep_indices]
    data['room'].y = data['room'].y[keep_indices]
    
    # Handle all edge types that involve room nodes
    for rel in data.edge_index_dict.keys():
        if 'room' in rel:  # Only process relations involving rooms
            edge_index = data[rel].edge_index
            edge_attr = data[rel].edge_attr
            
            # Determine which dimension contains room nodes
            room_dim = 0 if rel[0] == 'room' else 1
            
            # Create mask for edges to keep
            valid_edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
            for i in range(edge_index.size(1)):
                room_idx = edge_index[room_dim, i].item()
                if room_idx in remove_indices:
                    valid_edge_mask[i] = False
            
            # Apply mask to edges
            new_edge_index = edge_index[:, valid_edge_mask]
            if edge_attr is not None:
                new_edge_attr = edge_attr[valid_edge_mask]
            
            # Reindex room nodes in edge_index
            if room_dim == 0:
                new_edge_index[0] = torch.tensor(
                    [old_to_new[old.item()] for old in new_edge_index[0]], 
                    dtype=torch.long
                )
            else:
                new_edge_index[1] = torch.tensor(
                    [old_to_new[old.item()] for old in new_edge_index[1]], 
                    dtype=torch.long
                )
            
            # Update the edge data
            data[rel].edge_index = new_edge_index
            if edge_attr is not None:
                data[rel].edge_attr = new_edge_attr
    
    return data

def visualize_floorplan(data, title="Floorplan Visualization", ax=None):
    """
    Visualize floorplan graph with rooms and portals.
    If ax is provided, draws on that axis. Otherwise creates new figure.
    """
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add room nodes
    for i in range(data['room'].x.shape[0]):
        room_type = data['room'].y[i].item() if data['room'].y[i].item() != -1 else -1
        G.add_node(f"room_{i}", node_type='room', room_type=room_type)
    
    # Add portal nodes
    for i in range(data['portal'].x.shape[0]):
        G.add_node(f"portal_{i}", node_type='portal')
    
    # Add edges between rooms and portals
    if ('room', 'connects', 'portal') in data.edge_index_dict:
        for src, dst in data['room', 'connects', 'portal'].edge_index.t().tolist():
            G.add_edge(f"room_{src}", f"portal_{dst}")
    
    # Add edges between portals and rooms
    if ('portal', 'connects', 'room') in data.edge_index_dict:
        for src, dst in data['portal', 'connects', 'room'].edge_index.t().tolist():
            G.add_edge(f"portal_{src}", f"room_{dst}")
    
    # Generate layout using spring layout
    pos = nx.spring_layout(G, k=0.15, seed=42)
    
    # Set up node colors
    node_colors = []
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data['node_type'] == 'room':
            room_type = node_data['room_type']
            if room_type == 0:
                node_colors.append('lightgreen')  # Corridor
            elif room_type == 1:
                node_colors.append('lightblue')   # Toilet
            elif room_type == -1:
                node_colors.append('gray')        # Unknown/masked
            else:
                node_colors.append('salmon')      # Other
        else:
            node_colors.append('lightgray')       # Portal
            
            
    # Show edge weights if available ###################################################################
    edge_labels = {}
    
    if ('room', 'connects', 'portal') in data.edge_index_dict:
        for idx, (src, dst) in enumerate(data['room', 'connects', 'portal'].edge_index.t().tolist()):
            edge = (f"room_{src}", f"portal_{dst}")
            distance = data['room', 'connects', 'portal'].edge_attr[idx].item()
            edge_labels[edge] = f"{distance:.2f}"

    if ('portal', 'connects', 'room') in data.edge_index_dict:
        for idx, (src, dst) in enumerate(data['portal', 'connects', 'room'].edge_index.t().tolist()):
            edge = (f"portal_{src}", f"room_{dst}")
            distance = data['portal', 'connects', 'room'].edge_attr[idx].item()
            edge_labels[edge] = f"{distance:.2f}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    ########################################################################################################
    
    # Determine if we need to create our own figure
    own_figure = ax is None
    if own_figure:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
    
    # Draw the graph
    nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=80, ax=ax)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Corridor', markerfacecolor='lightgreen', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Toilet', markerfacecolor='lightblue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Other Room', markerfacecolor='salmon', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Portal', markerfacecolor='lightgray', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Masked Room', markerfacecolor='gray', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='best')
    ax.set_title(title)
    
    if own_figure:
        plt.tight_layout()
        return plt.gcf()



def visualize_masked_floorplan(data, title="Masked Floorplan", ax=None):
    """
    Visualize floorplan with masked nodes highlighted differently.
    If ax is provided, draws on that axis. Otherwise creates new figure.
    """
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add room nodes
    for i in range(data['room'].x.shape[0]):
        is_masked = False
        if hasattr(data['room'], 'masked_mask'):
            is_masked = data['room'].masked_mask[i].item()
        
        room_type = data['room'].y[i].item()
        if room_type == -1:
            is_masked = True
            # Get original type if available
            if hasattr(data['room'], 'y_original'):
                room_type = data['room'].y_original[i].item()
            
        G.add_node(f"room_{i}", 
                   node_type='room', 
                   room_type=room_type, 
                   is_masked=is_masked)
    
    # Add portal nodes
    for i in range(data['portal'].x.shape[0]):
        G.add_node(f"portal_{i}", node_type='portal')
    
    # Add edges between rooms and portals
    if ('room', 'connects', 'portal') in data.edge_index_dict:
        for src, dst in data['room', 'connects', 'portal'].edge_index.t().tolist():
            G.add_edge(f"room_{src}", f"portal_{dst}")
    
    # Add edges between portals and rooms
    if ('portal', 'connects', 'room') in data.edge_index_dict:
        for src, dst in data['portal', 'connects', 'room'].edge_index.t().tolist():
            G.add_edge(f"portal_{src}", f"room_{dst}")
    
    # Generate layout using spring layout
    pos = nx.spring_layout(G, k=0.15, seed=42)
    
    # Set up node colors and borders
    node_colors = []
    node_borders = []
    node_sizes = []
    
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data['node_type'] == 'room':
            room_type = node_data['room_type']
            is_masked = node_data['is_masked']
            
            if is_masked:
                # Show masked nodes with red border
                node_borders.append('red')
                node_sizes.append(120)  # Larger size for masked nodes
                
                # Color based on original type
                if room_type == 0:
                    node_colors.append('lightgreen')
                elif room_type == 1:
                    node_colors.append('lightblue')
                else:
                    node_colors.append('salmon')
            else:
                # Regular nodes
                if room_type == 0:
                    node_colors.append('lightgreen')
                elif room_type == 1:
                    node_colors.append('lightblue')
                else:
                    node_colors.append('salmon')
                node_borders.append('black')
                node_sizes.append(80)
        else:
            # Portal nodes
            node_colors.append('lightgray')
            node_borders.append('black')
            node_sizes.append(60)

    # Show edge weights if available ###################################################################
    edge_labels = {}
    
    if ('room', 'connects', 'portal') in data.edge_index_dict:
        for idx, (src, dst) in enumerate(data['room', 'connects', 'portal'].edge_index.t().tolist()):
            edge = (f"room_{src}", f"portal_{dst}")
            distance = data['room', 'connects', 'portal'].edge_attr[idx].item()
            edge_labels[edge] = f"{distance:.2f}"

    if ('portal', 'connects', 'room') in data.edge_index_dict:
        for idx, (src, dst) in enumerate(data['portal', 'connects', 'room'].edge_index.t().tolist()):
            edge = (f"portal_{src}", f"room_{dst}")
            distance = data['portal', 'connects', 'room'].edge_attr[idx].item()
            edge_labels[edge] = f"{distance:.2f}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    ########################################################################################################
    
    # Determine if we need to create our own figure
    own_figure = ax is None
    if own_figure:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
    
    # Draw the graph
    nx.draw(G, pos, node_color=node_colors, edgecolors=node_borders, 
            linewidths=1.5, node_size=node_sizes, with_labels=False, ax=ax)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Corridor', 
                   markerfacecolor='lightgreen', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Toilet', 
                   markerfacecolor='lightblue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Other Room', 
                   markerfacecolor='salmon', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Portal', 
                   markerfacecolor='lightgray', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Masked Node (For Prediction)', 
                   markerfacecolor='lightgreen', markeredgecolor='red', linewidth=2, markersize=12)
    ]
    ax.legend(handles=legend_elements, loc='best')
    ax.set_title(title)
    
    if own_figure:
        plt.tight_layout()
        return plt.gcf()



def visualize_masked_dataset(dataset, masked_dataset, idx=0):
    """
    Visualize a sample from the dataset before and after masking - NOTE BY MASKING I AM REFERRING TO REMOVING A NODE AND ITS ASSOCIATED EDGES
    Creates a single figure with two subplots.
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot original data
    visualize_floorplan(dataset[idx], title="Original Floorplan", ax=ax1)
    
    # Plot masked data
    visualize_masked_floorplan(masked_dataset[idx], title="Masked Floorplan (For Node Prediction)", ax=ax2)
    
    plt.tight_layout()
    plt.show()


def stats(dataset, masked_dataset, idx=0):
    """Calculate and print statistics about node removal/masking.
    
    Args:
        dataset: Original dataset (before removal/masking)
        masked_dataset: Dataset after removal/masking
        idx: Index of the sample to analyze
    """
    original_data = dataset[idx]
    masked_data = masked_dataset[idx]
    
    # Calculate statistics
    total_rooms = original_data['room'].num_nodes
    
    # Two ways to calculate removed rooms depending on approach:
    # 1. For masking approach (where y=-1 for masked nodes)
    if hasattr(masked_data['room'], 'y_original'):
        removed_rooms = (masked_data['room'].y == -1).sum().item()
    # 2. For removal approach (where nodes are actually deleted)
    else:
        removed_rooms = total_rooms - masked_data['room'].num_nodes
    
    removed_percentage = (removed_rooms / total_rooms) * 100
    
    # Print statistics
    print(f"Total rooms: {total_rooms}")
    print(f"Removed rooms: {removed_rooms} ({removed_percentage:.1f}%)")


# CODE ########################################################################################
# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Split the dataset (no masking yet)
train_dataset, val_dataset, test_dataset = random_split(
    dataset, 
    [2, 1, 1],  # Manual split since total is 4
    generator=torch.Generator().manual_seed(42)
)

# Mask validation and test datasets
remove_ratio = 0.3
val_masked = [remove_room_nodes(copy.deepcopy(data), remove_ratio) for data in val_dataset]
test_masked = [remove_room_nodes(copy.deepcopy(data), remove_ratio) for data in test_dataset]

# Create DataLoader objects for val_masked and test_masked'
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_masked, batch_size=1, shuffle=False)
test_loader = DataLoader(test_masked, batch_size=1, shuffle=False)
#########################################################################################

if __name__ == '__main__':
    if val_masked:
        idx = 0
        visualize_masked_dataset(dataset, val_masked, idx)
        stats(dataset, val_masked, idx)
        
    print("Train:", len(train_dataset))
    print("Val (masked):", len(val_masked))
    print("Test (masked):", len(test_masked))