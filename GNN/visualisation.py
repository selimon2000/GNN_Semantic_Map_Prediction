import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

import torch
import networkx as nx



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



def visualize_reconstructed_floorplan(original_data, reconstructed_data, title="Reconstructed Floorplan", ax=None):
    """
    Visualize the reconstructed floorplan with new nodes and edges highlighted.
    Uses maximized layout to ensure nodes are spread out as much as possible.
    """
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Original room count
    orig_room_count = original_data['room'].num_nodes
    
    # Add original room nodes
    for i in range(orig_room_count):
        room_type = original_data['room'].y[i].item() if original_data['room'].y[i].item() != -1 else -1
        G.add_node(f"room_{i}", node_type='room', room_type=room_type, is_original=True)
    
    # Add reconstructed room nodes (highlight new ones)
    for i in range(reconstructed_data['room'].num_nodes):
        if i >= orig_room_count:  # New node
            room_type = reconstructed_data['room'].y[i].item()
            G.add_node(f"room_{i}", node_type='room', room_type=room_type, is_original=False, is_new=True)
        else:  # Existing node
            room_type = reconstructed_data['room'].y[i].item()
            G.nodes[f"room_{i}"]["room_type"] = room_type  # Update in case it changed
    
    # Add portal nodes
    for i in range(reconstructed_data['portal'].num_nodes):
        G.add_node(f"portal_{i}", node_type='portal')
    
    # Add edges between rooms and portals (highlight new ones)
    if ('room', 'connects', 'portal') in reconstructed_data.edge_index_dict:
        for src, dst in reconstructed_data['room', 'connects', 'portal'].edge_index.t().tolist():
            is_new_edge = src >= orig_room_count  # Edge connected to new node
            G.add_edge(f"room_{src}", f"portal_{dst}", is_new=is_new_edge)
    
    # Add edges between portals and rooms
    if ('portal', 'connects', 'room') in reconstructed_data.edge_index_dict:
        for src, dst in reconstructed_data['portal', 'connects', 'room'].edge_index.t().tolist():
            is_new_edge = dst >= orig_room_count  # Edge connected to new node
            G.add_edge(f"portal_{src}", f"room_{dst}", is_new=is_new_edge)
    
    # MAXIMIZED LAYOUT: Combining multiple layout approaches for best spread
    
    # Start with a reasonable initial layout
    initial_pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)  # Much higher k value for more spread
    
    # Then apply Kamada-Kawai to refine it
    pos = nx.kamada_kawai_layout(G, pos=initial_pos)
    
    # Now expand the layout to fill the space
    # Find the current extent of the layout
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Scale factor to maximize the use of space
    # Higher scale makes the graph larger within the available space
    scale_factor = 4.0
    
    # Apply scaling to maximize space usage
    for node in pos:
        x, y = pos[node]
        # Normalize and then scale
        x_normalized = (x - min_x) / (max_x - min_x if max_x > min_x else 1)
        y_normalized = (y - min_y) / (max_y - min_y if max_y > min_y else 1)
        # Apply scaling and shift to center
        pos[node] = (
            (x_normalized * 2 - 1) * scale_factor,
            (y_normalized * 2 - 1) * scale_factor
        )
    
    # Apply node type positioning adjustment to prevent portal clustering
    portal_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'portal']
    room_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'room']
    
    # Further separate portals from each other to reduce congestion
    for i, portal1 in enumerate(portal_nodes):
        for portal2 in portal_nodes[i+1:]:
            # Calculate distance between portals
            x1, y1 = pos[portal1]
            x2, y2 = pos[portal2]
            dist = ((x1-x2)**2 + (y1-y2)**2)**0.5
            
            # If portals are too close, push them apart
            if dist < 0.5:
                # Direction from portal1 to portal2
                dx = (x2 - x1) / dist if dist > 0 else 0
                dy = (y2 - y1) / dist if dist > 0 else 0
                
                # Push apart, stronger when closer
                push = 0.2 * (0.5 - dist)
                pos[portal1] = (x1 - dx * push, y1 - dy * push)
                pos[portal2] = (x2 + dx * push, y2 + dy * push)
    
    # Move portals slightly towards their connected rooms
    for portal in portal_nodes:
        # Get connected room nodes
        neighbors = [n for n in G.neighbors(portal) if 'room_' in n]
        if neighbors:
            # Calculate the average position of connected rooms
            avg_x = sum(pos[neighbor][0] for neighbor in neighbors) / len(neighbors)
            avg_y = sum(pos[neighbor][1] for neighbor in neighbors) / len(neighbors)
            
            # Move portal 70% towards the average position of its neighbors
            x, y = pos[portal]
            pos[portal] = (
                0.3 * x + 0.7 * avg_x,
                0.3 * y + 0.7 * avg_y
            )
    
    # Set up node colors and styles
    node_colors = []
    node_borders = []
    node_sizes = []
    
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data['node_type'] == 'room':
            room_type = node_data['room_type']
            is_new = node_data.get('is_new', False)
            
            if is_new:
                # New nodes get special highlighting
                node_borders.append('lime')
                node_sizes.append(180)  # Larger to stand out more
                node_colors.append('lightgreen' if room_type == 0 else 
                                 'lightblue' if room_type == 1 else 
                                 'salmon')
            else:
                # Original nodes
                if room_type == 0:
                    node_colors.append('lightgreen')
                elif room_type == 1:
                    node_colors.append('lightblue')
                elif room_type == -1:
                    node_colors.append('gray')
                else:
                    node_colors.append('salmon')
                node_borders.append('black')
                node_sizes.append(150)  # Larger for better visibility
        else:
            # Portal nodes
            node_colors.append('lightgray')
            node_borders.append('black')
            node_sizes.append(100)  # Larger than before
    
    # Set up edge styles
    edge_colors = []
    edge_widths = []
    for u, v, data in G.edges(data=True):
        if data.get('is_new', False):
            edge_colors.append('lime')
            edge_widths.append(3.0/2)  # Make new edges more visible
        else:
            edge_colors.append('darkgray')  # Darker gray for better visibility
            edge_widths.append(1.5/2)
    
    # Show edge weights if available
    edge_labels = {}
    
    if ('room', 'connects', 'portal') in reconstructed_data.edge_index_dict:
        for idx, (src, dst) in enumerate(reconstructed_data['room', 'connects', 'portal'].edge_index.t().tolist()):
            edge = (f"room_{src}", f"portal_{dst}")
            if hasattr(reconstructed_data['room', 'connects', 'portal'], 'edge_attr'):
                distance = reconstructed_data['room', 'connects', 'portal'].edge_attr[idx].item()
                edge_labels[edge] = f"{distance:.2f}"
    
    if ('portal', 'connects', 'room') in reconstructed_data.edge_index_dict:
        for idx, (src, dst) in enumerate(reconstructed_data['portal', 'connects', 'room'].edge_index.t().tolist()):
            edge = (f"portal_{src}", f"room_{dst}")
            if hasattr(reconstructed_data['portal', 'connects', 'room'], 'edge_attr'):
                distance = reconstructed_data['portal', 'connects', 'room'].edge_attr[idx].item()
                edge_labels[edge] = f"{distance:.2f}"
    
    # Determine if we need to create our own figure
    own_figure = ax is None
    if own_figure:
        plt.figure(figsize=(16, 14))  # Much larger figure for maximum visibility
        ax = plt.gca()
    
    # Draw the graph
    nx.draw(G, pos, 
            node_color=node_colors, 
            edgecolors=node_borders,
            linewidths=2.0,
            node_size=node_sizes,
            edge_color=edge_colors,
            width=edge_widths,
            with_labels=False,
            ax=ax)
    
    # Add room labels with room types
    room_labels = {}
    for node in G.nodes():
        if 'room_' in node:
            node_data = G.nodes[node]
            if node_data['node_type'] == 'room':
                room_type = node_data['room_type']
                room_num = int(node.split('_')[1])
                if room_type == 0:
                    label = f"C{room_num}"  # Corridor with number
                elif room_type == 1:
                    label = f"T{room_num}"  # Toilet with number
                elif room_type == -1:
                    label = f"?{room_num}"  # Unknown with number
                else:
                    label = f"R{room_num}"  # Other Room with number
                room_labels[node] = label
    
    # Also add portal labels
    portal_labels = {}
    for node in G.nodes():
        if 'portal_' in node:
            portal_num = int(node.split('_')[1])
            portal_labels[node] = f"P{portal_num}"
    
    # Combine all labels
    all_labels = {**room_labels, **portal_labels}
    
    nx.draw_networkx_labels(G, pos, labels=all_labels, font_size=10, font_weight='bold', font_color='black', ax=ax)
    
    # Draw edge labels if available (with adjusted position to avoid overlapping with nodes)
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='blue', 
                                     label_pos=0.6, ax=ax)
    
    # Remove axis
    ax.set_axis_off()
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Corridor (C)', 
                   markerfacecolor='lightgreen', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Toilet (T)', 
                   markerfacecolor='lightblue', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Other Room (R)', 
                   markerfacecolor='salmon', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Portal (P)', 
                   markerfacecolor='lightgray', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Newly Added Room', 
                   markerfacecolor='lightgreen', markeredgecolor='lime', linewidth=2.5, markersize=14),
        plt.Line2D([0], [0], color='lime', lw=3.0, label='New Connections'),
        plt.Line2D([0], [0], color='darkgray', lw=1.5, label='Existing Connections')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    ax.set_title(title, fontsize=16)
    
    if own_figure:
        plt.tight_layout()
        return plt.gcf()
    
    
    
def visualize_masked_dataset(original_data, masked_data):
    """
    Visualize a sample before and after masking/removal.
    Creates a single figure with two subplots.
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot original data
    visualize_floorplan(original_data, title="Original Floorplan", ax=ax1)
    
    # Plot masked data
    visualize_masked_floorplan(masked_data, title="Masked Floorplan (For Node Prediction)", ax=ax2)
    
    plt.tight_layout()
    plt.show()



def visualize_masked_dataset_with_reconstructed(original_data, masked_data, reconstructed_data, sample_idx=0, total_samples=1):
    """
    Visualize a sample before and after masking/removal.
    Creates a single figure with three subplots.
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.subplots_adjust(bottom=0.2)  # Leave space for buttons if needed
    
    # Plot original data in the first subplot
    visualize_floorplan(original_data, 
                       title=f"Original Floorplan\nSample {sample_idx+1}/{total_samples}", 
                       ax=axes[0])
    
    # Plot masked data in the second subplot
    visualize_masked_floorplan(masked_data, 
                             title="Masked Floorplan", 
                             ax=axes[1])
    
    # Plot reconstructed data in the third subplot
    visualize_reconstructed_floorplan(masked_data,  # Note: This matches your working version
                                    reconstructed_data, 
                                    title="Reconstructed Floorplan", 
                                    ax=axes[2])
    
    plt.tight_layout()
    plt.show()