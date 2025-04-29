import random
import numpy as np
import copy

import torch
from torch_geometric.nn import HGTConv, HeteroConv
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Set random seeds for reproducibility, important to declare first as imported files will use this ##########
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#################################################

import sys
sys.path.append('/home/selimon/capstone_v3/GNN')
from split_dataset_new import *
from training_and_monitoring import *
# Visualisation functions gets imported from split_dataset


"""
- Answers: "How many nodes are missing from this graph?"
- Input: graph-level embedding
- Training: During training, this component learns to predict the difference between the
            node count in complete graphs and their incomplete counterparts.
- Architecture
  - First layer: Linear transformation followed by ReLU activation
  - Second layer: Linear transformation to output a single value
- Predicting too many nodes might create unrealistic structures, while predicting too few
  would leave the graph incomplete.
- When is this run: the model uses this prediction to determine exactly how many new node
  embeddings to generate with the NodeGenerator
  
- Learning of Structural Patterns
  - Can indirectly learn structural patterns based on graph embedding input
  - To increase characterisation of structural patterns and feature extraction add graph statistics
    - Degree Distribution
    - Clustering Coefficients
"""
class NodeExistencePredictor(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1))
    
    def forward(self, graph_embedding):
        """Predict number of missing nodes based on graph embedding"""
        # Output is positive real number
        return F.softplus(self.predictor(graph_embedding))



class GraphPooling(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.attention = torch.nn.Linear(hidden_channels, 1)
    
    def forward(self, x_dict):
        """Create graph-level embedding by pooling node features"""
        graph_embedding = None
        
        for node_type, x in x_dict.items():
            # Attention-based pooling
            attn_scores = self.attention(x).sigmoid()
            node_type_emb = (x * attn_scores).sum(dim=0)
            
            if graph_embedding is None:
                graph_embedding = node_type_emb
            else:
                graph_embedding += node_type_emb
                
        return graph_embedding



class NodeGenerator(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Generator network
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels * 2, hidden_channels))
        
        # Conditioner network (conditions on graph structure)
        self.conditioner = torch.nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        
    def forward(self, graph_embedding, num_nodes_to_generate, existing_nodes=None):
        """Generate embeddings for new nodes"""
        # Initialize with graph embedding
        h0 = graph_embedding.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, hidden_channels]
        
        # If we have existing nodes, condition on them
        if existing_nodes is not None and len(existing_nodes) > 0:
            existing_nodes_batch = existing_nodes.unsqueeze(0)
            _, h_n = self.conditioner(existing_nodes_batch, h0)
            h0 = h_n  # Updated hidden state
        
        # Generate new nodes
        noise = torch.randn(num_nodes_to_generate, self.hidden_channels, 
                          device=graph_embedding.device) * 0.1
        last_hidden = h0.squeeze(0).squeeze(0)  # Shape: [hidden_channels]
        generated = self.generator(noise + last_hidden)
        
        return generated



class ReconstructionHGTModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        
        # Feature encoding
        self.encoders = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.encoders[node_type] = torch.nn.Linear(1, hidden_channels)
        
        # HGT convolutions
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        # Node type classifier
        self.node_classifier = torch.nn.Linear(hidden_channels, out_channels)
        
        # Link prediction
        self.link_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid())
        
        # Edge weight prediction
        self.edge_weight_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid())
        
        # Graph-level pooling
        self.graph_pooling = GraphPooling(hidden_channels)
        
        # Node existence predictor
        self.node_existence = NodeExistencePredictor(hidden_channels)
        
        # Improved node generator
        self.node_generator = NodeGenerator(hidden_channels)
        
        # Position predictor for new nodes
        self.position_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 2))
        
        # Edge existence classifier
        self.edge_existence = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid())

    """
    - Creates node-level embeddings for each node in the graph
    - Creates h_dict, a dictionary mapping node types to their embeddings
    - Iterates through multiple graph convolution layers (self.convs), where layers refers to hops
    - Graph_pooling aggregates these node embeddings into a single graph-level representation
    - Node-level embeddings encode both:
      - The inherent attributes of each node (initial features like room type)
      - The structural role of the node within the graph (its connections and neighborhood)
    """
    def encode_graph(self, x_dict, edge_index_dict):
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.encoders[node_type](x)

        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            for node_type in h_dict:
                h_dict[node_type] = F.relu(h_dict[node_type])
        
        return h_dict
    
    def forward(self, x_dict, edge_index_dict, edge_candidates=None):
        # Encode graph
        h_dict = self.encode_graph(x_dict, edge_index_dict)
        
        # Node classification
        node_preds = self.node_classifier(h_dict['room'])
        
        # CREATE GRAPH-LEVEL REPRESENTATION BY COMBINING THE NODE-LEVEL EMBEDDINGS USING ATTENTION-BASED POOLING
        graph_embedding = self.graph_pooling(h_dict)
        
        # Predict number of missing nodes
        missing_nodes_estimate = self.node_existence(graph_embedding)
        
        # Link prediction (if edge candidates provided)
        link_preds, weight_preds = None, None
        if edge_candidates is not None:
            link_preds, weight_preds = self.predict_links_and_weights(h_dict, edge_candidates)
        
        return node_preds, link_preds, weight_preds, missing_nodes_estimate, h_dict, graph_embedding
    
    def predict_links_and_weights(self, h_dict, edge_candidates):
        """Predict edge existence and weights"""
        link_scores = []
        weight_preds = []
        
        for r_idx, p_idx in edge_candidates:
            if r_idx < h_dict['room'].size(0) and p_idx < h_dict['portal'].size(0):
                r_emb = h_dict['room'][r_idx]
                p_emb = h_dict['portal'][p_idx]
                pair_embed = torch.cat([r_emb, p_emb], dim=0)
                link_score = self.link_predictor(pair_embed)
                link_scores.append(link_score)
                weight = self.edge_weight_predictor(pair_embed)
                weight_preds.append(weight)
        
        if link_scores:
            return torch.stack(link_scores).flatten(), torch.stack(weight_preds).flatten()
        else:
            return torch.tensor([], device=h_dict['room'].device), torch.tensor([], device=h_dict['room'].device)
    
    """
    Takes the graph embedding as a starting point
    Conditions on existing nodes to ensure compatibility
    Generates unique embeddings for each predicted missing node
    """
    def generate_missing_nodes(self, h_dict, graph_embedding, num_missing):
        """Generate embeddings for missing nodes"""
        device = graph_embedding.device
        num_missing = int(num_missing.item())
        
        if num_missing <= 0:
            return torch.tensor([], device=device)
        
        existing_rooms = h_dict['room']
        generated_embeddings = self.node_generator(
            graph_embedding, 
            num_missing,
            existing_rooms)
        
        return generated_embeddings
    
    def predict_node_positions(self, new_embeddings):
        """Predict positions for new nodes relative to existing ones"""
        positions = self.position_predictor(new_embeddings)
        positions = torch.sigmoid(positions)
        return positions
    
    def reconstruct_graph(self, data, missing_ratio=0.3):
        """Reconstruct a graph with missing nodes"""
        device = data['room'].x.device
        x_dict = {node_type: x for node_type, x in data.x_dict.items()}
        edge_index_dict = data.edge_index_dict
        
        _, _, _, missing_nodes_estimate, h_dict, graph_embedding = self(x_dict, edge_index_dict)
        expected_missing = int(data['room'].num_nodes * missing_ratio / (1 - missing_ratio))
        num_missing = max(1, min(int(missing_nodes_estimate.item()), expected_missing * 2))
        
        generated_embeddings = self.generate_missing_nodes(
            h_dict, 
            graph_embedding, 
            torch.tensor([num_missing], device=device))
        
        if len(generated_embeddings) == 0:
            return data
        
        room_types = self.node_classifier(generated_embeddings).argmax(dim=1)
        new_positions = self.predict_node_positions(generated_embeddings)
        feature_dim = data['room'].x.size(1)
        new_features = torch.zeros((num_missing, feature_dim), device=device)
        
        new_x_dict = {k: v for k, v in x_dict.items()}
        new_x_dict['room'] = torch.cat([x_dict['room'], new_features], dim=0)
        h_dict['room'] = torch.cat([h_dict['room'], generated_embeddings], dim=0)
        
        new_room_offset = data['room'].num_nodes
        edge_candidates = []
        
        for i in range(num_missing):
            new_idx = new_room_offset + i
            for p_idx in range(data['portal'].num_nodes):
                edge_candidates.append((new_idx, p_idx))
                
        if not edge_candidates:
            reconstructed_data = copy.deepcopy(data)
            reconstructed_data['room'].x = new_x_dict['room']
            reconstructed_data['room'].y = torch.cat([data['room'].y, room_types], dim=0)
            reconstructed_data['room'].pos = torch.cat([
                data['room'].pos if hasattr(data['room'], 'pos') else 
                torch.zeros((data['room'].num_nodes, 2), device=device),
                new_positions], dim=0)
            return reconstructed_data
        
        edge_candidates = torch.tensor(edge_candidates, device=device)
        link_preds, weight_preds = self.predict_links_and_weights(h_dict, edge_candidates)
        threshold = 0.5 + (torch.rand(1, device=device).item() * 0.2 - 0.1)
        valid_edges = edge_candidates[link_preds > threshold]
        valid_weights = weight_preds[link_preds > threshold]
        
        reconstructed_data = copy.deepcopy(data)
        reconstructed_data['room'].x = new_x_dict['room']
        reconstructed_data['room'].y = torch.cat([data['room'].y, room_types], dim=0)
        if hasattr(data['room'], 'pos'):
            reconstructed_data['room'].pos = torch.cat([
                data['room'].pos,
                new_positions], dim=0)
        else:
            reconstructed_data['room'].pos = torch.cat([
                torch.zeros((data['room'].num_nodes, 2), device=device),
                new_positions], dim=0)
        
        if len(valid_edges) > 0:
            new_edges = torch.stack([valid_edges[:, 0], valid_edges[:, 1]], dim=0)
            
            if ('room', 'connects', 'portal') in reconstructed_data.edge_index_dict:
                reconstructed_data[('room', 'connects', 'portal')].edge_index = torch.cat([
                    reconstructed_data[('room', 'connects', 'portal')].edge_index,
                    new_edges], dim=1)
                
                if hasattr(reconstructed_data[('room', 'connects', 'portal')], 'edge_attr'):
                    reconstructed_data[('room', 'connects', 'portal')].edge_attr = torch.cat([
                        reconstructed_data[('room', 'connects', 'portal')].edge_attr,
                        valid_weights.view(-1, 1)], dim=0)
            else:
                reconstructed_data[('room', 'connects', 'portal')].edge_index = new_edges
                reconstructed_data[('room', 'connects', 'portal')].edge_attr = valid_weights.view(-1, 1)
            
            if ('portal', 'connects', 'room') in reconstructed_data.edge_index_dict:
                reversed_edges = torch.stack([new_edges[1], new_edges[0]], dim=0)
                reconstructed_data[('portal', 'connects', 'room')].edge_index = torch.cat([
                    reconstructed_data[('portal', 'connects', 'room')].edge_index,
                    reversed_edges], dim=1)
                
                if hasattr(reconstructed_data[('portal', 'connects', 'room')], 'edge_attr'):
                    reconstructed_data[('portal', 'connects', 'room')].edge_attr = torch.cat([
                        reconstructed_data[('portal', 'connects', 'room')].edge_attr,
                        valid_weights.view(-1, 1)], dim=0)
        
        return reconstructed_data



def generate_edge_candidates_with_weights(data, neg_ratio=1.0):
    """Generate positive and negative edge candidates for link prediction with edge weights."""
    # Check if the edge type exists
    if ('room', 'connects', 'portal') not in data.edge_index_dict:
        device = data['room'].x.device if 'room' in data.x_dict else 'cpu'
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.empty(0, device=device),
            torch.empty(0, device=device)
        )
    
    pos_edges = data.edge_index_dict[('room', 'connects', 'portal')].t()
    device = pos_edges.device
    
    if hasattr(data[('room', 'connects', 'portal')], 'edge_attr') and data[('room', 'connects', 'portal')].edge_attr is not None:
        pos_weights = data[('room', 'connects', 'portal')].edge_attr
    else:
        pos_weights = torch.ones(pos_edges.size(0), 1, device=device)
    
    num_rooms = data['room'].num_nodes
    num_portals = data['portal'].num_nodes
    num_neg = int(neg_ratio * len(pos_edges))
    
    neg_edges = torch.stack([
        torch.randint(0, num_rooms, (num_neg,), device=device),
        torch.randint(0, num_portals, (num_neg,), device=device)], dim=1)
    
    neg_weights = torch.zeros(neg_edges.size(0), 1, device=device)
    edge_candidates = torch.cat([pos_edges, neg_edges], dim=0)
    edge_labels = torch.cat([
        torch.ones(len(pos_edges), device=device),
        torch.zeros(len(neg_edges), device=device)])
    edge_weights = torch.cat([pos_weights, neg_weights], dim=0)
    
    perm = torch.randperm(len(edge_candidates), device=device)
    return edge_candidates[perm], edge_labels[perm], edge_weights[perm]



def evaluate_reconstruction(model, loader, device='cpu'):
    """Evaluate model with node reconstruction metrics"""
    model.eval()
    metrics = {
        'node_f1': 0,
        'node_acc': 0,
        'link_auc': 0,
        'existence_mae': 0,
        'reconstruction_acc': 0,
        'reconstruction_iou': 0
    }
    
    samples_processed = 0
    
    with torch.no_grad():
        for data in loader:
            # Skip invalid data
            if isinstance(data, list):
                continue
                
            try:
                # Check if data has required attributes
                if not hasattr(data, 'x_dict') or not hasattr(data, 'edge_index_dict'):
                    continue
                    
                orig_data = copy.deepcopy(data).to(device)
                masked_data = remove_room_nodes(copy.deepcopy(data), remove_ratio=0.3).to(device)
                
                # Skip if masking removed all rooms
                if masked_data['room'].num_nodes == 0:
                    continue
                    
                true_missing = orig_data['room'].num_nodes - masked_data['room'].num_nodes
                
                # Ensure we have edge_index_dict
                if not hasattr(masked_data, 'edge_index_dict'):
                    masked_data.edge_index_dict = {}
                
                reconstructed_data = model.reconstruct_graph(masked_data, missing_ratio=0.3)
                
                # Node classification metrics
                if reconstructed_data['room'].num_nodes > masked_data['room'].num_nodes:
                    remaining_node_count = masked_data['room'].num_nodes
                    pred_types = reconstructed_data['room'].y[remaining_node_count:].cpu().numpy()
                    true_types = orig_data['room'].y[remaining_node_count:orig_data['room'].num_nodes].cpu().numpy()
                    
                    if len(pred_types) > 0 and len(true_types) > 0:
                        min_len = min(len(pred_types), len(true_types))
                        metrics['node_acc'] += accuracy_score(true_types[:min_len], pred_types[:min_len])
                        metrics['node_f1'] += f1_score(true_types[:min_len], pred_types[:min_len], 
                                                      average='weighted', zero_division=0)
                
                # Missing nodes estimation
                _, _, _, missing_nodes_estimate, _, _ = model(
                    {k: v for k, v in masked_data.x_dict.items()}, 
                    masked_data.edge_index_dict)
                metrics['existence_mae'] += abs(missing_nodes_estimate.item() - true_missing)
                
                # Check if required edge type exists in reconstructed data
                if ('room', 'connects', 'portal') in orig_data.edge_index_dict:
                    # Original edges
                    orig_edges = set()
                    edges = orig_data[('room', 'connects', 'portal')].edge_index.t().cpu().numpy()
                    for e in edges:
                        orig_edges.add(tuple(e))
                    
                    # Reconstructed edges - CHECK IF THE EDGE TYPE EXISTS
                    recon_edges = set()
                    if ('room', 'connects', 'portal') in reconstructed_data.edge_index_dict:
                        edges = reconstructed_data[('room', 'connects', 'portal')].edge_index.t().cpu().numpy()
                        for e in edges:
                            recon_edges.add(tuple(e))
                    
                    # Calculate edge metrics
                    true_positives = len(orig_edges & recon_edges)
                    false_positives = len(recon_edges - orig_edges)
                    false_negatives = len(orig_edges - recon_edges)
                    
                    precision = true_positives / (true_positives + false_positives + 1e-10)
                    recall = true_positives / (true_positives + false_negatives + 1e-10)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                    
                    metrics['reconstruction_acc'] += f1
                
                # Node count metrics
                orig_nodes = orig_data['room'].num_nodes
                recon_nodes = reconstructed_data['room'].num_nodes
                metrics['reconstruction_iou'] += min(orig_nodes, recon_nodes) / max(orig_nodes, recon_nodes)
                
                samples_processed += 1
                
            except Exception as e:
                print(f"Validation error: {str(e)}")
                continue
    
    # Normalize metrics
    for key in metrics:
        if samples_processed > 0:
            metrics[key] /= samples_processed
        else:
            metrics[key] = 0.0
    
    return metrics



def load_model_for_inference(model_path, metadata, device='cpu'):
    """
    Load a trained model for inference.
    
    Args:
        model_path: Path to the saved model state dict
        metadata: Graph metadata (node types, edge types)
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        Loaded model ready for inference
    """
    # Initialize the model with the same architecture
    model = ReconstructionHGTModel(
        hidden_channels=128,
        out_channels=len(ROOM_TYPE_MAP.keys()),
        num_heads=8,
        num_layers=3,
        metadata=metadata
    )
    
    # Load the saved state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model



# Update the run_inference function to include statistics
def run_inference_with_stats(model, data, original_data=None, missing_ratio=0.3, device='cpu'):
    """
    Run inference with the loaded model on new data and get statistics.
    
    Args:
        model: Loaded model
        data: Input data to reconstruct
        original_data: Original complete data (if available)
        missing_ratio: Expected ratio of missing nodes
        device: Device to run inference on
        
    Returns:
        Tuple of (reconstructed_data, statistics)
    """
    # Make sure data is on the correct device
    data = data.to(device)
    if original_data is not None:
        original_data = original_data.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Reconstruct the graph
        reconstructed_data = model.reconstruct_graph(data, missing_ratio=missing_ratio)
        
        # Get statistics
        stats = get_reconstruction_statistics(original_data, data, reconstructed_data)
        
    return reconstructed_data, stats



def get_reconstruction_statistics(original_data, masked_data, reconstructed_data):
    """
    Calculate statistics about the graph reconstruction process.
    
    Args:
        original_data: The original complete graph data (if available)
        masked_data: The input data with missing nodes
        reconstructed_data: The reconstructed graph data
        
    Returns:
        Dictionary containing various statistics about the reconstruction
    """
    stats = {}
    
    # Node counts
    masked_node_count = masked_data['room'].num_nodes
    recon_node_count = reconstructed_data['room'].num_nodes
    stats['original_nodes'] = original_data['room'].num_nodes if original_data is not None else 'N/A'
    stats['masked_nodes'] = masked_node_count
    stats['reconstructed_nodes'] = recon_node_count
    stats['new_nodes_created'] = recon_node_count - masked_node_count
    
    # Node types
    if hasattr(reconstructed_data['room'], 'y'):
        # Get node types for masked data
        masked_types = {}
        for i in range(masked_data['room'].num_nodes):
            node_type = masked_data['room'].y[i].item()
            masked_types[node_type] = masked_types.get(node_type, 0) + 1
        
        # Get node types for reconstructed data
        recon_types = {}
        for i in range(reconstructed_data['room'].num_nodes):
            node_type = reconstructed_data['room'].y[i].item()
            recon_types[node_type] = recon_types.get(node_type, 0) + 1
        
        # Calculate new nodes by type
        new_nodes_by_type = {}
        for node_type in recon_types:
            masked_count = masked_types.get(node_type, 0)
            recon_count = recon_types.get(node_type, 0)
            new_count = max(0, recon_count - masked_count)
            if new_count > 0:
                new_nodes_by_type[node_type] = new_count
        
        stats['masked_node_types'] = masked_types
        stats['reconstructed_node_types'] = recon_types
        stats['new_nodes_by_type'] = new_nodes_by_type
        
        # If we have a room type map, we can translate the numeric types to names
        if 'ROOM_TYPE_MAP' in globals():
            inv_room_type_map = {v: k for k, v in ROOM_TYPE_MAP.items()}
            new_nodes_by_type_name = {}
            for node_type, count in new_nodes_by_type.items():
                type_name = inv_room_type_map.get(node_type, f"Type_{node_type}")
                new_nodes_by_type_name[type_name] = count
            stats['new_nodes_by_type_name'] = new_nodes_by_type_name
    
    # Edge counts
    masked_edge_count = 0
    recon_edge_count = 0
    
    if ('room', 'connects', 'portal') in masked_data.edge_index_dict:
        masked_edge_count = masked_data[('room', 'connects', 'portal')].edge_index.size(1)
    
    if ('room', 'connects', 'portal') in reconstructed_data.edge_index_dict:
        recon_edge_count = reconstructed_data[('room', 'connects', 'portal')].edge_index.size(1)
    
    stats['masked_edges'] = masked_edge_count
    stats['reconstructed_edges'] = recon_edge_count
    stats['new_edges_created'] = recon_edge_count - masked_edge_count
    
    # Edge connectivity of new nodes
    if ('room', 'connects', 'portal') in reconstructed_data.edge_index_dict and stats['new_nodes_created'] > 0:
        edge_index = reconstructed_data[('room', 'connects', 'portal')].edge_index
        new_node_start_idx = masked_node_count
        
        # Count edges connected to new nodes
        new_node_edges = 0
        for i in range(edge_index.size(1)):
            if edge_index[0, i] >= new_node_start_idx:
                new_node_edges += 1
        
        stats['edges_connected_to_new_nodes'] = new_node_edges
        stats['avg_edges_per_new_node'] = new_node_edges / stats['new_nodes_created'] if stats['new_nodes_created'] > 0 else 0
    
    # Calculate structural changes
    if original_data is not None and ('room', 'connects', 'portal') in original_data.edge_index_dict:
        # Calculate accuracy of reconstruction compared to original
        orig_edges = set()
        edges = original_data[('room', 'connects', 'portal')].edge_index.t().cpu().numpy()
        for e in edges:
            orig_edges.add(tuple(e))
        
        recon_edges = set()
        if ('room', 'connects', 'portal') in reconstructed_data.edge_index_dict:
            edges = reconstructed_data[('room', 'connects', 'portal')].edge_index.t().cpu().numpy()
            for e in edges:
                recon_edges.add(tuple(e))
        
        # Edge metrics
        true_positives = len(orig_edges & recon_edges)
        false_positives = len(recon_edges - orig_edges)
        false_negatives = len(orig_edges - recon_edges)
        
        stats['edge_true_positives'] = true_positives
        stats['edge_false_positives'] = false_positives
        stats['edge_false_negatives'] = false_negatives
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        stats['edge_precision'] = precision
        stats['edge_recall'] = recall
        stats['edge_f1'] = f1
        
        # Node count accuracy
        orig_nodes = original_data['room'].num_nodes
        recon_nodes = reconstructed_data['room'].num_nodes
        stats['node_count_accuracy'] = min(orig_nodes, recon_nodes) / max(orig_nodes, recon_nodes)
    
    return stats



def print_reconstruction_statistics(stats):
    """
    Print the reconstruction statistics in a readable format.
    
    Args:
        stats: Dictionary of statistics from get_reconstruction_statistics
    """
    print("\n===== RECONSTRUCTION STATISTICS =====")
    print(f"Original nodes: {stats['original_nodes']}")
    print(f"Masked nodes: {stats['masked_nodes']}")
    print(f"Reconstructed nodes: {stats['reconstructed_nodes']}")
    print(f"New nodes created: {stats['new_nodes_created']}")
    
    if 'new_nodes_by_type_name' in stats:
        print("\nNew nodes by type:")
        for type_name, count in stats['new_nodes_by_type_name'].items():
            print(f"  - {type_name}: {count}")
    elif 'new_nodes_by_type' in stats:
        print("\nNew nodes by type:")
        for type_id, count in stats['new_nodes_by_type'].items():
            print(f"  - Type {type_id}: {count}")
    
    print(f"\nMasked edges: {stats['masked_edges']}")
    print(f"Reconstructed edges: {stats['reconstructed_edges']}")
    print(f"New edges created: {stats['new_edges_created']}")
    
    if 'edges_connected_to_new_nodes' in stats:
        print(f"Edges connected to new nodes: {stats['edges_connected_to_new_nodes']}")
        print(f"Average edges per new node: {stats['avg_edges_per_new_node']:.2f}")
    
    if 'edge_precision' in stats:
        print("\nReconstruction accuracy:")
        print(f"Edge precision: {stats['edge_precision']:.4f}")
        print(f"Edge recall: {stats['edge_recall']:.4f}")
        print(f"Edge F1 score: {stats['edge_f1']:.4f}")
        print(f"Node count accuracy: {stats['node_count_accuracy']:.4f}")
    
    print("=====================================\n")



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    sample_data = dataset[0]
    metadata = (list(sample_data.x_dict.keys()), list(sample_data.edge_index_dict.keys()))

    main_train(device, metadata)
    # main_test(device, metadata, 15)