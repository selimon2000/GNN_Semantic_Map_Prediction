import os
from pathlib import Path
import math
from pathlib import Path
from glob import glob
# XML Specific Modules
import xml.etree.ElementTree as ET

# Machine learning specific modules
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch_geometric
from torch_geometric.data import Dataset, HeteroData



# Variables to be exported ########################################
ROOM_TYPE_MAP = {
    # Corridor
    'KORRIDOR': 0,
    # Toilet
    'Toalett': 1,
    'TOALETT': 1
}
# Else it equals to 'unknown' - Later in Code:
# space_type = space.get('type', 'unknown')
REVERSE_ROOM_TYPE_MAP = {v: k for k, v in ROOM_TYPE_MAP.items()}


    
def visualize_floorplan_circular(dataset, idx):

    # Get the HeteroData object
    data = dataset[idx]
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Extract positions from the data or generate positions
    # Since the original code doesn't store actual positions, we'll generate them
    
    # First, create positions for rooms in a circle layout
    room_count = data['room'].x.shape[0]
    room_positions = {}
    
    # Arrange rooms in a circle
    for i in range(room_count):
        angle = 2 * math.pi * i / room_count
        radius = 10
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        room_positions[i] = [x, y]
        
        # Add room nodes with positions
        G.add_node(f"room_{i}",
                   pos=[x, y],
                   node_type='room',
                   room_type=data['room'].y[i].item() if 'y' in data['room'] else -1)
    
    # Now place portals between connected rooms
    portal_positions = {}
    
    # First, gather all room-portal connections
    room_to_portal_connections = {}
    if ('room', 'connects', 'portal') in data.edge_index_dict:
        edge_index = data['room', 'connects', 'portal'].edge_index
        for src, dst in edge_index.t().tolist():
            if dst not in room_to_portal_connections:
                room_to_portal_connections[dst] = []
            room_to_portal_connections[dst].append(src)
    
    # Gather all portal-room connections
    portal_to_room_connections = {}
    if ('portal', 'connects', 'room') in data.edge_index_dict:
        edge_index = data['portal', 'connects', 'room'].edge_index
        for src, dst in edge_index.t().tolist():
            if src not in portal_to_room_connections:
                portal_to_room_connections[src] = []
            portal_to_room_connections[src].append(dst)
    
    # Place portals between their connected rooms
    for portal_idx in range(data['portal'].x.shape[0]):
        connected_room_indices = []
        
        # Get rooms that connect to this portal
        if portal_idx in room_to_portal_connections:
            connected_room_indices.extend(room_to_portal_connections[portal_idx])
        
        # Get rooms that this portal connects to
        if portal_idx in portal_to_room_connections:
            connected_room_indices.extend(portal_to_room_connections[portal_idx])
        
        # If portal has connections, place it between the rooms
        if connected_room_indices:
            avg_x = sum(room_positions[room_idx][0] for room_idx in connected_room_indices) / len(connected_room_indices)
            avg_y = sum(room_positions[room_idx][1] for room_idx in connected_room_indices) / len(connected_room_indices)
            
            # Move portal slightly toward center to avoid overlap
            center_pull = 0.3
            avg_x = avg_x * (1 - center_pull)
            avg_y = avg_y * (1 - center_pull)
            
            portal_positions[portal_idx] = [avg_x, avg_y]
            
            # Add portal node with position
            G.add_node(f"portal_{portal_idx}",
                       pos=[avg_x, avg_y],
                       node_type='portal')
    
    # Helper function to compute Euclidean distance
    def get_distance(n1, n2):
        p1 = G.nodes[n1]['pos']
        p2 = G.nodes[n2]['pos']
        return round(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2), 2)
    
    # Add edges with weights from room -> portal
    if ('room', 'connects', 'portal') in data.edge_index_dict:
        for src, dst in data['room', 'connects', 'portal'].edge_index.t().tolist():
            src_node = f"room_{src}"
            dst_node = f"portal_{dst}"
            weight = get_distance(src_node, dst_node)
            G.add_edge(src_node, dst_node, weight=weight)
    
    # Add edges with weights from portal -> room
    if ('portal', 'connects', 'room') in data.edge_index_dict:
        for src, dst in data['portal', 'connects', 'room'].edge_index.t().tolist():
            src_node = f"portal_{src}"
            dst_node = f"room_{dst}"
            weight = get_distance(src_node, dst_node)
            G.add_edge(src_node, dst_node, weight=weight)
    
    # Get positions for drawing
    pos = nx.get_node_attributes(G, 'pos')
    
    # Color setup
    node_colors = []
    room_type_labels = {}
    
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data['node_type'] == 'room':
            room_type = node_data['room_type']
            if room_type == 0:
                node_colors.append('lightgreen')
                room_type_labels[node] = 'Corridor'
            elif room_type == 1:
                node_colors.append('lightblue')
                room_type_labels[node] = 'Toilet'
            else:
                node_colors.append('salmon')
                room_type_labels[node] = 'Other'
        else:
            node_colors.append('gray')
            room_type_labels[node] = 'Portal'
    
    # Draw nodes
    plt.figure(figsize=(22, 10), dpi=200)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.7, alpha=0.35, edge_color='black')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=room_type_labels, font_size=5)
    
    # Draw edge weights (distances)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=4, label_pos=0.55)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Corridor', markerfacecolor='lightgreen', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Toilet', markerfacecolor='lightblue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Other Room', markerfacecolor='salmon', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Portal', markerfacecolor='gray', markersize=10)
    ]
    
    plt.legend(handles=legend_elements, loc='best')
    plt.title(f"Floorplan Visualization - {data.building_name if hasattr(data, 'building_name') else 'Unknown'}")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()



def visualize_floorplan_spring(dataset, idx):
    # Get the HeteroData object
    data = dataset[idx]
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add room nodes
    for i in range(data['room'].x.shape[0]):
        G.add_node(f"room_{i}",
                   node_type='room',
                   room_type=data['room'].y[i].item() if 'y' in data['room'] else -1)
    
    # Add portal nodes
    for i in range(data['portal'].x.shape[0]):
        G.add_node(f"portal_{i}",
                   node_type='portal')
    
    # Add edges between rooms and portals
    if ('room', 'connects', 'portal') in data.edge_index_dict:
        for src, dst in data['room', 'connects', 'portal'].edge_index.t().tolist():
            src_node = f"room_{src}"
            dst_node = f"portal_{dst}"
            
            # Use edge_attr if available, otherwise default weight
            if hasattr(data['room', 'connects', 'portal'], 'edge_attr'):
                idx = ((data['room', 'connects', 'portal'].edge_index[0] == src) & 
                       (data['room', 'connects', 'portal'].edge_index[1] == dst)).nonzero().item()
                weight = data['room', 'connects', 'portal'].edge_attr[idx].item()
            else:
                weight = 1.0
                
            G.add_edge(src_node, dst_node, weight=weight)
    
    # Add edges between portals and rooms
    if ('portal', 'connects', 'room') in data.edge_index_dict:
        for src, dst in data['portal', 'connects', 'room'].edge_index.t().tolist():
            src_node = f"portal_{src}"
            dst_node = f"room_{dst}"
            
            # Use edge_attr if available, otherwise default weight
            if hasattr(data['portal', 'connects', 'room'], 'edge_attr'):
                idx = ((data['portal', 'connects', 'room'].edge_index[0] == src) & 
                       (data['portal', 'connects', 'room'].edge_index[1] == dst)).nonzero().item()
                weight = data['portal', 'connects', 'room'].edge_attr[idx].item()
            else:
                weight = 1.0
                
            G.add_edge(src_node, dst_node, weight=weight)
    
    # Generate layout using spring layout
    # The k parameter affects the spacing (increase for more space between nodes)
    # The weight parameter uses the edge weights we defined (distances)
    pos = nx.spring_layout(G, k=0.15, seed=42, weight='weight')
    
    # Helper function to compute distance between nodes in the layout
    def get_layout_distance(n1, n2):
        p1 = pos[n1]
        p2 = pos[n2]
        return round(math.sqrt(sum((p1[i] - p2[i])**2 for i in range(len(p1)))), 2)
    
    # Update edge weights based on layout distances
    for u, v in G.edges():
        G[u][v]['weight'] = get_layout_distance(u, v)
    
    # Color setup
    node_colors = []
    room_type_labels = {}
    
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data['node_type'] == 'room':
            room_type = node_data['room_type']
            if room_type == 0:
                node_colors.append('lightgreen')
                room_type_labels[node] = 'Corridor'
            elif room_type == 1:
                node_colors.append('lightblue')
                room_type_labels[node] = 'Toilet'
            else:
                node_colors.append('salmon')
                room_type_labels[node] = 'Other'
        else:
            node_colors.append('gray')
            room_type_labels[node] = 'Portal'
    
    # Draw the graph
    plt.figure(figsize=(22, 10), dpi=160)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=90, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.7, alpha=0.35, edge_color='black')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=room_type_labels, font_size=5)
    
    # Draw edge weights (distances)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=4, label_pos=0.55)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Corridor', markerfacecolor='lightgreen', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Toilet', markerfacecolor='lightblue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Other Room', markerfacecolor='salmon', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Portal', markerfacecolor='gray', markersize=10)
    ]
    
    plt.legend(handles=legend_elements, loc='best')
    plt.title(f"Floorplan Visualization - {data.building_name if hasattr(data, 'building_name') else 'Unknown'}")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


    
class FloorPlanDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, log=True):
        root = str(Path(root).resolve())
        self.building_dirs = [d for d in glob(os.path.join(root, '*')) if os.path.isdir(d)]
        super().__init__(root, transform, pre_transform, pre_filter, log=log)
    
    @property
    def raw_dir(self):
        # Since each building has its own raw directory, we'll handle this differently
        return self.root  # The root contains building directories
    
    @property
    def processed_dir(self):
        # Each building has its own processed directory
        return self.root  # We'll handle processed paths per building
    
    @property
    def raw_file_names(self):
        # Get all XML files from all building raw directories
        xml_files = []
        for building_dir in self.building_dirs:
            raw_dir = os.path.join(building_dir, 'raw')
            if os.path.exists(raw_dir):
                xml_files.extend([
                    os.path.join(building_dir, 'raw', f) 
                    for f in os.listdir(raw_dir) 
                    if f.endswith('.xml')
                ])
        return xml_files
    
    @property
    def processed_file_names(self):
        # Create processed filenames that maintain building directory structure
        processed_files = []
        for raw_path in self.raw_file_names:
            building_dir = os.path.dirname(os.path.dirname(raw_path))  # Get building directory
            filename = os.path.basename(raw_path).replace('.xml', '.pt')
            processed_files.append(os.path.join(building_dir, 'processed', filename))
        return processed_files
    
    def process(self):
        # Process each XML file in all building directories
        for raw_path in self.raw_file_names:
            # Set up paths for this building
            building_dir = os.path.dirname(os.path.dirname(raw_path))
            processed_dir = os.path.join(building_dir, 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            
            filename = os.path.basename(raw_path)
            processed_path = os.path.join(processed_dir, filename.replace('.xml', '.pt'))
            
            try:
                # Parse XML file
                tree = ET.parse(raw_path)
                root = tree.getroot()
                
                # Get scale information
                scale_element = root.find('Scale')
                if scale_element is None:
                    print(f"Skipping {raw_path}: No scale information found")
                    continue
                    
                pixel_distance = float(scale_element.get('PixelDistance'))
                real_distance = float(scale_element.get('RealDistance'))
                
                def convert_to_real_distance(distance):
                    return (distance / pixel_distance) * real_distance
                
                # Initialize data structures
                pyg_data = HeteroData()
                
                # Collect all rooms and portals
                room_features = []  # Will store room types only
                room_labels = []
                room_names = []
                room_positions = []  # Temporary storage for distance calculations
                
                portal_features = []  # Will store portal types only
                portal_positions = []  # Temporary storage for distance calculations
                portal_targets = []  # Keep track of which room each portal targets
                
                edge_index_room_to_portal = []
                edge_index_portal_to_room = []
                edge_attr_room_portal = []  # Distance between rooms and portals
                edge_attr_portal_room = []  # Distance between portals and rooms
                
                # First pass: collect all rooms
                for space in root.findall('space'):
                    space_name = space.get('name')
                    space_type = space.get('type', 'unknown')
                    space_type_encoded = ROOM_TYPE_MAP.get(space_type, 2)
                    
                    centroid = space.find('contour/centroid')
                    if centroid is None:
                        print(f"Skipping space {space_name}: No centroid found")
                        continue
                        
                    x, y = float(centroid.get('x')), float(centroid.get('y'))
                    real_x, real_y = convert_to_real_distance(x), convert_to_real_distance(y)
                    
                    room_features.append(torch.tensor([space_type_encoded], dtype=torch.float))
                    room_labels.append(space_type_encoded)
                    room_names.append(space_name)
                    room_positions.append((real_x, real_y))
                
                # Second pass: collect all portals
                for space in root.findall('space'):
                    space_name = space.get('name')
                    
                    for linesegment in space.findall('contour/linesegment'):
                        if linesegment.get('type') == 'Portal':
                            target = linesegment.get('target')
                            if target is None:
                                print(f"Skipping portal in {space_name}: No target specified")
                                continue
                                
                            x1, y1 = float(linesegment.get('x1')), float(linesegment.get('y1'))
                            x2, y2 = float(linesegment.get('x2')), float(linesegment.get('y2'))
                            portal_x, portal_y = convert_to_real_distance((x1 + x2) / 2), convert_to_real_distance((y1 + y2) / 2)
                            
                            portal_features.append(torch.tensor([0], dtype=torch.float))  # 0 represents portal type
                            portal_positions.append((portal_x, portal_y))
                            portal_targets.append(target)
                            
                            # Connect source room to portal
                            try:
                                source_idx = room_names.index(space_name)
                                edge_index_room_to_portal.append([source_idx, len(portal_positions) - 1])
                                
                                # Calculate distance between room and portal
                                room_x, room_y = room_positions[source_idx]
                                distance = math.sqrt((room_x - portal_x)**2 + (room_y - portal_y)**2)
                                edge_attr_room_portal.append(torch.tensor([distance], dtype=torch.float))
                            except ValueError:
                                print(f"Warning: Portal source room {space_name} not found in room list")
                
                # Connect portals to target rooms and calculate distances
                for portal_idx, target in enumerate(portal_targets):
                    if target in room_names:
                        try:
                            target_idx = room_names.index(target)
                            edge_index_portal_to_room.append([portal_idx, target_idx])
                            
                            # Calculate distance between portal and target room
                            portal_x, portal_y = portal_positions[portal_idx]
                            room_x, room_y = room_positions[target_idx]
                            distance = math.sqrt((portal_x - room_x)**2 + (portal_y - room_y)**2)
                            edge_attr_portal_room.append(torch.tensor([distance], dtype=torch.float))
                        except ValueError:
                            print(f"Warning: Portal target room {target} not found in room list")
                
                # Build the heterogeneous graph
                if room_features:
                    pyg_data['room'].x = torch.stack(room_features)  # Room types only
                    pyg_data['room'].y = torch.tensor(room_labels, dtype=torch.long)
                
                if portal_features:
                    pyg_data['portal'].x = torch.stack(portal_features)  # Portal types only
                
                # Add edges with distance attributes
                if edge_index_room_to_portal:
                    pyg_data['room', 'connects', 'portal'].edge_index = torch.tensor(
                        edge_index_room_to_portal, dtype=torch.long).t().contiguous()
                    pyg_data['room', 'connects', 'portal'].edge_attr = torch.stack(edge_attr_room_portal)
                
                if edge_index_portal_to_room:
                    pyg_data['portal', 'connects', 'room'].edge_index = torch.tensor(
                        edge_index_portal_to_room, dtype=torch.long).t().contiguous()
                    pyg_data['portal', 'connects', 'room'].edge_attr = torch.stack(edge_attr_portal_room)
                
                # Add graph-level attributes
                pyg_data.num_rooms = len(room_features)
                pyg_data.num_portals = len(portal_features)
                pyg_data.building_name = f"{os.path.basename(building_dir)}_{filename.split('.')[0]}"
                
                # Apply transformations if specified
                if self.pre_transform is not None:
                    pyg_data = self.pre_transform(pyg_data)
                
                # Skip if filtered out
                if self.pre_filter is not None and not self.pre_filter(pyg_data):
                    continue
                
                # Save processed data to the building's processed directory
                torch.save(pyg_data, processed_path)
                
            except ET.ParseError as e:
                print(f"XML parsing error in {raw_path}: {str(e)}")
                continue
            except Exception as e:
                print(f"Unexpected error processing {raw_path}: {str(e)}")
                continue
    
    def len(self):
        # Count all existing processed files across all buildings
        count = 0
        for building_dir in self.building_dirs:
            processed_dir = os.path.join(building_dir, 'processed')
            if os.path.exists(processed_dir):
                count += len([f for f in os.listdir(processed_dir) if f.endswith('.pt')])
        return count
    
    def get(self, idx):
        if idx < 0 or idx >= self.len():
            raise IndexError(f"Index {idx} out of range")
        
        # Collect all processed files across all buildings
        processed_files = []
        for building_dir in self.building_dirs:
            processed_dir = os.path.join(building_dir, 'processed')
            if os.path.exists(processed_dir):
                processed_files.extend([
                    os.path.join(processed_dir, f)
                    for f in os.listdir(processed_dir)
                    if f.endswith('.pt')
                ])
        
        processed_path = processed_files[idx]
        data = torch.load(processed_path)
        return data


# Initialize dataset with the root directory containing all subdirectories
dataset = FloorPlanDataset(root='/home/selimon/capstone_v3/dataset')

# Remove any graphs in the dataset which are not instances of HeteroData - not necessary anymore now that I've found the root of the issue
# dataset = [d for d in dataset if isinstance(d, HeteroData)]

if __name__ == '__main__':   
    # You can now visualize any floorplan in any subdirectory by its index
    idx = 0  # First file across all subdirectories
    visualize_floorplan_spring(dataset, idx=idx)
    visualize_floorplan_circular(dataset, idx=idx)
    print(f"Total floorplans in dataset: {len(dataset)}")
    
    # visualize_floorplan_spring(dataset, idx=idx)
    # visualize_floorplan_circular(dataset, idx=idx)
    print(len(dataset))
    
    
# if __name__ == '__main__':
#     # Sifting through list
#     for i, d in enumerate(dataset):
#         if not isinstance(d, HeteroData):
#             print(f"dataset[{i}] type: {type(d)}")
#             # visualize_floorplan_spring(dataset, i)
    
    
# if __name__ == '__main__':
#     # Get all processed files (same logic as in get())
#     processed_files = []
#     for building_dir in dataset.building_dirs:
#         processed_dir = os.path.join(building_dir, 'processed')
#         if os.path.exists(processed_dir):
#             processed_files.extend([
#                 os.path.join(processed_dir, f)
#                 for f in os.listdir(processed_dir)
#                 if f.endswith('.pt')
#             ])

#     # Check each file
#     for idx, path in enumerate(processed_files):
#         data = torch.load(path)
#         if isinstance(data, str):
#             print(f"Corrupted file: {path}")
#             # Map back to raw XML
#             xml_file = os.path.basename(path).replace('.pt', '.xml')
#             building_dir = os.path.dirname(os.path.dirname(path))
#             xml_path = os.path.join(building_dir, 'raw', xml_file)
#             print(f"Source XML file: {xml_path}")
#             print(f"String content: {data}")