import sys
sys.path.append('/home/selimon/capstone_v3/GNN')
from data_set import *
from visualisation import *

import random
import copy

import torch
from torch.utils.data import Subset, random_split
import networkx as nx
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split



def remove_room_nodes(data, remove_ratio=0.5):
    # First check if the data has room nodes at all
    if 'room' not in data.node_types if hasattr(data, 'node_types') else data.x_dict.keys():
        return data
    
    # Now safely check the number of room nodes
    if data['room'].num_nodes == 0:
        return data

    device = data['room'].x.device
    
    # Calculate degrees - both incoming and outgoing
    degrees = torch.zeros(data['room'].num_nodes, dtype=torch.long, device=device)
    
    # Use edge_types if available, otherwise use edge_index_dict keys
    edge_types = data.edge_types if hasattr(data, 'edge_types') else data.edge_index_dict.keys()
    
    # Count outgoing edges (room -> portal)
    if ('room', 'connects', 'portal') in edge_types:
        src = data[('room', 'connects', 'portal')].edge_index[0]
        degrees.scatter_add_(0, src, torch.ones_like(src))
    
    # Count incoming edges (portal -> room)
    if ('portal', 'connects', 'room') in edge_types:
        dst = data[('portal', 'connects', 'room')].edge_index[1]
        degrees.scatter_add_(0, dst, torch.ones_like(dst))
    
    # Sort nodes by degree (ascending) to remove least connected first
    sorted_indices = torch.argsort(degrees)
    num_to_remove = max(1, int(remove_ratio * data['room'].num_nodes))
    remove_indices = sorted_indices[:num_to_remove]
    keep_indices = sorted_indices[num_to_remove:]
    
    # Create mapping from old to new indices
    old_to_new = {old.item(): new for new, old in enumerate(keep_indices)}
    
    # Filter room nodes
    data['room'].x = data['room'].x[keep_indices]
    if hasattr(data['room'], 'y'):
        data['room'].y = data['room'].y[keep_indices]
    
    # Update graph metadata if exists
    if hasattr(data, 'num_rooms'):
        data.num_rooms = len(keep_indices)

    # Process all edges involving rooms
    for edge_type in edge_types:
        edge_data = data[edge_type]
        edge_index = edge_data.edge_index
        edge_attr = getattr(edge_data, 'edge_attr', None)
        
        # Check if rooms are involved in this edge type
        is_source = edge_type[0] == 'room'
        is_target = edge_type[2] == 'room'
        
        if is_source or is_target:
            # Create mask for valid edges
            mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
            
            if is_source:
                source_nodes = edge_index[0]
                mask &= ~torch.isin(source_nodes, remove_indices)
            
            if is_target:
                target_nodes = edge_index[1]
                mask &= ~torch.isin(target_nodes, remove_indices)
            
            # Apply mask and update edges
            new_edge_index = edge_index[:, mask]
            data[edge_type].edge_index = new_edge_index
            
            # Remap room indices
            if is_source:
                new_edge_index[0] = torch.tensor(
                    [old_to_new[old.item()] for old in new_edge_index[0]],
                    device=device
                )
            if is_target:
                new_edge_index[1] = torch.tensor(
                    [old_to_new[old.item()] for old in new_edge_index[1]],
                    device=device
                )
            
            # Update edge attributes if they exist
            if edge_attr is not None:
                data[edge_type].edge_attr = edge_attr[mask]

    return data



def split_dataset(dataset, train_ratio, val_ratio, test_ratio):
    """Split dataset into train/val/test while maintaining class balance"""
    assert abs(1 - (train_ratio + val_ratio + test_ratio)) < 1e-8, "Ratios must sum to 1"

    # Get labels if they exist
    labels = []
    for data in dataset:
        if hasattr(data, 'y'):
            labels.append(data.y.item() if data.y.numel() == 1 else data.y)
        elif hasattr(data, 'room') and hasattr(data['room'], 'y'):
            labels.append(data['room'].y[0].item() if data['room'].y.numel() == 1 else data['room'].y)
        else:
            labels.append(0)  # Default label if none exists

    # First split into train+val and test
    train_val_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=test_ratio,
        random_state=42,
        stratify=labels,
        shuffle=True
    )

    # Then split train_val into train and val
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio/(train_ratio+val_ratio),
        random_state=42,
        stratify=[labels[i] for i in train_val_indices],
        shuffle=True
    )

    return train_indices, val_indices, test_indices



def stats(original_data, masked_data):
    """Print statistics about the original and masked data"""

    total_rooms = original_data['room'].num_nodes
    remaining_rooms = masked_data['room'].num_nodes
    removed_rooms = total_rooms - remaining_rooms

    print(f"Total rooms: {total_rooms}")
    print(f"Remaining rooms: {remaining_rooms}")
    print(f"Removed rooms: {removed_rooms} ({removed_rooms/total_rooms*100:.1f}%)")
    print("---")
    
    
    
# Split the dataset
train_indices, val_indices, test_indices = split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

# APPLY NODE REMOVAL WITH CONSISTENT REMOVE_RATIO
remove_ratio = 0.10

# Generate masked datasets
train_masked = [remove_room_nodes(copy.deepcopy(dataset[i]), remove_ratio) for i in train_indices]
val_masked = [remove_room_nodes(copy.deepcopy(dataset[i]), remove_ratio) for i in val_indices]
test_masked = [remove_room_nodes(copy.deepcopy(dataset[i]), remove_ratio) for i in test_indices]

# Create DataLoaders
  # Original 
train_loader = DataLoader([dataset[i] for i in train_indices], batch_size=1, shuffle=True)
val_loader = DataLoader([dataset[i] for i in val_indices], batch_size=1, shuffle=False)
test_loader = DataLoader([dataset[i] for i in test_indices], batch_size=1, shuffle=False)
  # Masked
train_masked_loader = DataLoader(train_masked, batch_size=1, shuffle=True)
val_masked_loader = DataLoader(val_masked, batch_size=1, shuffle=False)
test_masked_loader = DataLoader(test_masked, batch_size=1, shuffle=False)



if __name__ == '__main__':
    # # General Statistics
    # print(f"Total dataset size: {len(dataset)}")
    # print(f"Train size: {len(train_indices)}")
    # print(f"Val size: {len(val_indices)}")
    # print(f"Test size: {len(test_indices)}")
    
    # # Specific Statistics
    # print("\n=== Training Split ===")
    # for i, orig_idx in enumerate(train_indices[:3]):
    #     print(f"Sample {i} (Original Index {orig_idx}):")
    #     stats(dataset[orig_idx], train_masked[i])
    #     visualize_masked_dataset(dataset[orig_idx], train_masked[i])

    # print("\n=== Validation Split ===")
    # for i, orig_idx in enumerate(val_indices[:3]):
    #     print(f"Sample {i} (Original Index {orig_idx}):")
    #     stats(dataset[orig_idx], val_masked[i])
    #     visualize_masked_dataset(dataset[orig_idx], val_masked[i])

    # print("\n=== Test Split ===")
    # for i, orig_idx in enumerate(test_indices[:3]):
    #     print(f"Sample {i} (Original Index {orig_idx}):")
    #     stats(dataset[orig_idx], test_masked[i])
    #     visualize_masked_dataset(dataset[orig_idx], test_masked[i])
    
    """         
    # DOUBLE CHECKING
    for batch_idx, (original_batch, masked_batch) in enumerate(zip(train_loader, train_masked_loader)):
        print(f"Type of original_batch: {type(original_batch)}")
        print(f"Type of masked_batch: {type(masked_batch)}")
        # If it's a list, check the first element
        if isinstance(original_batch, list):
            print(f"Length of original_batch: {len(original_batch)}")
            print(f"Type of original_batch[0]: {type(original_batch[0])}")
    
    for batch_idx, (original_batch, masked_batch) in enumerate(zip(val_loader, val_masked_loader)):
        print(f"Type of original_batch: {type(original_batch)}")
        print(f"Type of masked_batch: {type(masked_batch)}")
        # If it's a list, check the first element
        if isinstance(original_batch, list):
            print(f"Length of original_batch: {len(original_batch)}")
            print(f"Type of original_batch[0]: {type(original_batch[0])}")

    for batch_idx, (original_batch, masked_batch) in enumerate(zip(test_loader, test_masked_loader)):
        print(f"Type of original_batch: {type(original_batch)}")
        print(f"Type of masked_batch: {type(masked_batch)}")
        # If it's a list, check the first element
        if isinstance(original_batch, list):
            print(f"Length of original_batch: {len(original_batch)}")
            print(f"Type of original_batch[0]: {type(original_batch[0])}")
    """