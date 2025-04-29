split_dataset:

from data_set import *
from visualisation import *



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




graph_completion.py:
    
from split_dataset import *

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



def train_reconstruction_model(model, train_loader, train_masked_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """Train model with node reconstruction objective using pre-masked data"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    node_criterion = torch.nn.CrossEntropyLoss()
    link_criterion = torch.nn.BCELoss()
    weight_criterion = torch.nn.MSELoss()
    existence_criterion = torch.nn.L1Loss()
    
    model = model.to(device)
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        node_loss_total = 0
        link_loss_total = 0
        weight_loss_total = 0
        existence_loss_total = 0
        
        batches_processed = 0

        # Iterate through both original and masked data loaders together
        for original_batch, masked_batch in zip(train_loader, train_masked_loader):
            # Skip invalid batches
            if isinstance(original_batch, list) or isinstance(masked_batch, list):
                print(f"Skipping invalid batch: original={type(original_batch)}, masked={type(masked_batch)}")
                continue
            
            try:
                # Try to move data to device
                original_data = original_batch.to(device)
                masked_data = masked_batch.to(device)
                
                optimizer.zero_grad()
                
                # Calculate ground truth for missing nodes count
                true_missing = original_data['room'].num_nodes - masked_data['room'].num_nodes
                
                # Forward pass on masked data
                x_dict = {node_type: x for node_type, x in masked_data.x_dict.items()}
                edge_index_dict = masked_data.edge_index_dict
                
                # Generate edge candidates with weights
                edge_candidates, edge_labels, edge_weights = generate_edge_candidates_with_weights(masked_data)
                
                # Forward pass
                node_preds, link_preds, weight_preds, missing_nodes_estimate, h_dict, _ = model(
                    x_dict, edge_index_dict, edge_candidates)
                
                # Calculate losses
                mask = masked_data['room'].y != -1
                node_loss = node_criterion(node_preds[mask], masked_data['room'].y[mask])
                
                link_loss = link_criterion(link_preds, edge_labels.float()) if len(link_preds) > 0 else 0
                
                pos_mask = edge_labels == 1
                weight_loss = weight_criterion(weight_preds[pos_mask].view(-1, 1), edge_weights[pos_mask]) if torch.any(pos_mask) else 0
                
                existence_loss = existence_criterion(missing_nodes_estimate, torch.tensor([true_missing], device=device).float())
                
                # Combined loss
                loss = node_loss + 0.7 * link_loss + 0.5 * weight_loss + 1.5 * existence_loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track losses
                total_loss += loss.item()
                node_loss_total += node_loss.item()
                link_loss_total += link_loss.item() if isinstance(link_loss, torch.Tensor) else link_loss
                weight_loss_total += weight_loss.item() if isinstance(weight_loss, torch.Tensor) else weight_loss
                existence_loss_total += existence_loss.item()
                
                batches_processed += 1
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
        
        # Adjust average calculations based on actually processed batches
        if batches_processed > 0:
            total_loss /= batches_processed
            node_loss_total /= batches_processed
            link_loss_total /= batches_processed
            weight_loss_total /= batches_processed
            existence_loss_total /= batches_processed
        
        scheduler.step()
        
        # Validation (only if we've processed at least some batches)
        if batches_processed > 0 and (epoch % 5 == 0 or epoch == epochs - 1):
            val_metrics = evaluate_reconstruction(model, val_loader, device)
            
            print(f"Epoch: {epoch:03d}, Loss: {total_loss:.4f} "
                  f"(Node: {node_loss_total:.4f}, "
                  f"Link: {link_loss_total:.4f}, "
                  f"Weight: {weight_loss_total:.4f}, "
                  f"Existence: {existence_loss_total:.4f})")
            print(f"Val Node F1: {val_metrics['node_f1']:.4f}, "
                  f"Val Link AUC: {val_metrics['link_auc']:.4f}, "
                  f"Val Existence MAE: {val_metrics['existence_mae']:.4f}, "
                  f"Val Recon Accuracy: {val_metrics['reconstruction_acc']:.4f}")
            
            if val_metrics['node_f1'] > best_val_f1:
                best_val_f1 = val_metrics['node_f1']
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, 'best_reconstruction_model.pt')
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model



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
                
                # Edge reconstruction metrics
                if ('room', 'connects', 'portal') in orig_data.edge_index_dict:
                    # Original edges
                    orig_edges = set()
                    edges = orig_data[('room', 'connects', 'portal')].edge_index.t().cpu().numpy()
                    for e in edges:
                        orig_edges.add(tuple(e))
                    
                    # Reconstructed edges
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



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
 
    # Get dataset metadata
    sample_data = dataset[0]
    metadata = (list(sample_data.x_dict.keys()), list(sample_data.edge_index_dict.keys()))
      
    # Define reconstruction model
    model = ReconstructionHGTModel( hidden_channels=128,
                                    out_channels=len(ROOM_TYPE_MAP.keys()),
                                    num_heads=8,
                                    num_layers=3,
                                    metadata=metadata)

    # Train model
    model = train_reconstruction_model(
        model, 
        train_loader,
        train_masked_loader,
        val_loader, 
        epochs = 25,
        lr = 0.001, 
        device = device)
    
    # Evaluate on test set
    test_metrics = evaluate_reconstruction(model, test_loader, device=device)
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")