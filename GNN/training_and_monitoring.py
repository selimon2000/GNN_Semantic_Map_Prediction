import random
import numpy as np
import copy

import torch
from torch_geometric.nn import HGTConv, HeteroConv
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# Set random seeds for reproducibility ##########
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
from graph_completion import *



def train_reconstruction_model_with_monitoring(model, train_loader, train_masked_loader, val_loader, 
                                              dataset, val_indices, epochs=100, lr=0.001, 
                                              device='cpu', patience=10, monitor_frequency=5):
    
    # Initialize metrics dictionaries with empty lists
    train_metrics = {
        'epoch': [],
        'loss': [],
        'node_loss': [],
        'link_loss': [],
        'weight_loss': [],
        'existence_loss': [],
        'node_f1': [],
        'node_acc': [],
        'link_auc': [],
        'existence_mae': [],
        'reconstruction_acc': [],
        'reconstruction_iou': []
    }
    
    val_metrics = {
        'epoch': [],
        'loss': [],
        'node_f1': [],
        'node_acc': [],
        'link_auc': [],
        'existence_mae': [],
        'reconstruction_acc': [],
        'reconstruction_iou': []
    }

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
        
        # Record training metrics for this epoch
        train_metrics['epoch'].append(epoch)
        train_metrics['loss'].append(total_loss)
        train_metrics['node_loss'].append(node_loss_total)
        train_metrics['link_loss'].append(link_loss_total)
        train_metrics['weight_loss'].append(weight_loss_total)
        train_metrics['existence_loss'].append(existence_loss_total)
        
        # Validation (only if we've processed at least some batches)
        if batches_processed > 0 and (epoch % 5 == 0 or epoch == epochs - 1):
            evaluation_results = evaluate_reconstruction(model, val_loader, device)
            
            # Update validation metrics with results from evaluation
            val_metrics['epoch'].append(epoch)
            val_metrics['loss'].append(evaluation_results.get('loss', 0))
            val_metrics['node_f1'].append(evaluation_results['node_f1'])
            val_metrics['node_acc'].append(evaluation_results['node_acc'])
            val_metrics['link_auc'].append(evaluation_results['link_auc'])
            val_metrics['existence_mae'].append(evaluation_results['existence_mae'])
            val_metrics['reconstruction_acc'].append(evaluation_results['reconstruction_acc'])
            val_metrics['reconstruction_iou'].append(evaluation_results['reconstruction_iou'])
            
            # Also update training metrics that come from validation
            train_metrics['node_f1'].append(evaluation_results['node_f1'])
            train_metrics['node_acc'].append(evaluation_results['node_acc'])
            train_metrics['link_auc'].append(evaluation_results['link_auc'])
            train_metrics['existence_mae'].append(evaluation_results['existence_mae'])
            train_metrics['reconstruction_acc'].append(evaluation_results['reconstruction_acc'])
            train_metrics['reconstruction_iou'].append(evaluation_results['reconstruction_iou'])
            
            print(f"Epoch: {epoch:03d}, Loss: {total_loss:.4f} "
                  f"(Node: {node_loss_total:.4f}, "
                  f"Link: {link_loss_total:.4f}, "
                  f"Weight: {weight_loss_total:.4f}, "
                  f"Existence: {existence_loss_total:.4f})")
            print(f"Val Node F1: {evaluation_results['node_f1']:.4f}, "
                  f"Val Link AUC: {evaluation_results['link_auc']:.4f}, "
                  f"Val Existence MAE: {evaluation_results['existence_mae']:.4f}, "
                  f"Val Recon Accuracy: {evaluation_results['reconstruction_acc']:.4f}")
            
            if evaluation_results['node_f1'] > best_val_f1:
                best_val_f1 = evaluation_results['node_f1']
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, 'best_reconstruction_model.pt')
                
        # Just adding the monitoring part to be integrated with the previous function
        if epoch % monitor_frequency == 0 or epoch == epochs - 1:
            print(f"Monitoring training progress at epoch {epoch}...")
            monitor_training_progress(model, val_loader, dataset, val_indices, epoch, device=device)
        
        # # Optionally plot during training
        # if epoch % monitor_frequency == 0:
        #     plot_training_curves(train_metrics, val_metrics)
        plot_training_curves(train_metrics, val_metrics)
        
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # After training completes
    plot_training_curves(train_metrics, val_metrics)
    
    return model
   
    

def visualize_node_generation_analysis(analysis, original_data, save_path=None):
    """
    Visualize the node generation analysis results.
    Args:
        analysis: Results from analyze_node_generation
        original_data: Original input data
        save_path: Path to save visualization
    """
    plt.figure(figsize=(16, 12))
    
    # 1. Visualize node embeddings with PCA
    plt.subplot(2, 2, 1)
    # Get room embeddings
    if 'generated_embeddings' in analysis and analysis['generated_embeddings'] is not None:
        # Apply PCA to reduce dimensionality for visualization
        room_types = analysis['original_node_types']['room']
        generated_types = analysis['generated_types']
        
        # Combine existing and generated embeddings
        if 'room' in original_data.x_dict:
            existing_room_embeddings = analysis['graph_embedding']
            all_embeddings = np.concatenate([
                [existing_room_embeddings],  # Single graph embedding
                analysis['generated_embeddings']
            ], axis=0)
            
            # Apply PCA
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(all_embeddings)
            
            # Plot existing graph embedding
            plt.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1],
                        color='black', marker='*', s=200, label='Graph Embedding')
            
            # Plot generated embeddings
            # Create a colormap
            cmap = plt.cm.tab10
            
            for i in range(1, len(embeddings_2d)):
                node_type = generated_types[i-1]
                color = cmap(node_type % 10)
                plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                           color=color, marker='o', s=100,
                           label=f'Type {node_type}' if i == 1 else "")
                           
            plt.title('PCA of Graph Embedding and Generated Node Embeddings')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No generated embeddings',
                    horizontalalignment='center', verticalalignment='center')
    
    # 2. Attention weights for graph pooling
    plt.subplot(2, 2, 2)
    if 'attention_weights' in analysis:
        for node_type, weights in analysis['attention_weights'].items():
            plt.bar(node_type, weights.mean(), yerr=weights.std(), capsize=10)
        plt.title('Average Attention Weights by Node Type')
        plt.ylabel('Attention Weight')
        plt.grid(True)
    
    # 3. Distribution of generated node types
    plt.subplot(2, 2, 3)
    if 'generated_types' in analysis and len(analysis['generated_types']) > 0:
        # Count occurrences of each node type
        unique_types, counts = np.unique(analysis['generated_types'], return_counts=True)
        # Create a list of colors from the colormap
        colors = [plt.cm.tab10(t % 10) for t in unique_types]
        plt.bar([f'Type {t}' for t in unique_types], counts, color=colors)
        plt.title('Distribution of Generated Node Types')
        plt.ylabel('Count')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No node types generated',
                horizontalalignment='center', verticalalignment='center')
    
    # 4. Missing nodes estimate
    plt.subplot(2, 2, 4)
    plt.bar(['Estimated Missing Nodes'], [analysis['missing_nodes_estimate']], color='purple')
    plt.title('Estimated Number of Missing Nodes')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Analysis visualization saved to {save_path}")
    else:
        plt.show()



def monitor_training_progress(model, val_loader, dataset, val_indices, epoch, device='cpu'):
    """
    Monitor the training progress periodically by analyzing a sample.
    
    Args:
        model: Current model state
        val_loader: Validation data loader
        dataset: Original dataset
        val_indices: Validation indices in the dataset
        epoch: Current training epoch
        device: Device to run monitoring on
    """
    import os
    os.makedirs("training_progress", exist_ok=True)
    
    # Choose a sample from validation set for monitoring
    for i, data in enumerate(val_loader):
        # Skip invalid batches
        if isinstance(data, list):
            continue
            
        try:
            # Get original data
            val_idx = val_indices[i] if hasattr(val_loader.dataset, 'indices') else i
            original_data = dataset[val_idx]
            
            # Create masked version
            masked_data = remove_room_nodes(copy.deepcopy(data), remove_ratio=0.3)
            
            # Skip if masking removed all rooms
            if masked_data['room'].num_nodes == 0:
                continue
                
            # Analyze node generation
            analysis = analyze_node_generation(model, masked_data, device=device)
            
            # Visualize analysis
            save_path = f"training_progress/epoch_{epoch:03d}_analysis.png"
            visualize_node_generation_analysis(analysis, original_data, save_path=save_path)
            
            # Run reconstruction and visualize
            reconstructed_data, _ = run_inference_with_stats(
                model, masked_data, original_data=original_data, device=device)
            
            vis_save_path = f"training_progress/epoch_{epoch:03d}_reconstruction.png"
            visualize_reconstruction_comparison( original_data, masked_data, reconstructed_data, save_path=vis_save_path)
            
            # Only do one sample
            break
            
        except Exception as e:
            print(f"Error during monitoring: {str(e)}")
            continue
    
    
    
def visualize_reconstruction_comparison(original_data, masked_data, reconstructed_data, save_path=None):
    """
    Visualize the original, masked, and reconstructed graphs side by side.
    
    Args:
        original_data: The original complete graph data
        masked_data: The input data with missing nodes
        reconstructed_data: The reconstructed graph data
        save_path: Path to save the visualization (if None, display only)
    """

    
    plt.figure(figsize=(18, 6))
    
    # Create a helper function to convert heterogeneous graph to networkx
    def convert_to_nx(data):
        G = nx.Graph()
        
        # Add room nodes
        for i in range(data['room'].num_nodes):
            # Use node type and position for visualization
            node_type = data['room'].y[i].item() if hasattr(data['room'], 'y') else -1
            G.add_node(f'r{i}', type='room', room_type=node_type)
        
        # Add portal nodes
        for i in range(data['portal'].num_nodes):
            G.add_node(f'p{i}', type='portal')
        
        # Add edges
        if ('room', 'connects', 'portal') in data.edge_index_dict:
            edge_index = data[('room', 'connects', 'portal')].edge_index
            for i in range(edge_index.size(1)):
                room_idx = edge_index[0, i].item()
                portal_idx = edge_index[1, i].item()
                G.add_edge(f'r{room_idx}', f'p{portal_idx}')
        
        return G
    
    # Convert graphs to networkx
    original_nx = convert_to_nx(original_data)
    masked_nx = convert_to_nx(masked_data)
    recon_nx = convert_to_nx(reconstructed_data)
    
    # Set up node colors based on type
    def get_node_colors(G):
        colors = []
        for node in G.nodes():
            if G.nodes[node]['type'] == 'room':
                # Use different colors for different room types
                room_type = G.nodes[node]['room_type']
                colors.append(plt.cm.tab10(room_type % 10))
            else:
                # Portals are gray
                colors.append('gray')
        return colors
    
    # Plot the graphs
    plt.subplot(1, 3, 1)
    pos_original = nx.spring_layout(original_nx, seed=42)
    nx.draw(original_nx, pos_original, node_color=get_node_colors(original_nx), 
            with_labels=True, node_size=300, font_size=8)
    plt.title(f"Original Graph\n({original_data['room'].num_nodes} rooms, {original_data['portal'].num_nodes} portals)")
    
    plt.subplot(1, 3, 2)
    pos_masked = nx.spring_layout(masked_nx, seed=42)
    nx.draw(masked_nx, pos_masked, node_color=get_node_colors(masked_nx), 
            with_labels=True, node_size=300, font_size=8)
    plt.title(f"Masked Graph\n({masked_data['room'].num_nodes} rooms, {masked_data['portal'].num_nodes} portals)")
    
    plt.subplot(1, 3, 3)
    pos_recon = nx.spring_layout(recon_nx, seed=42)
    nx.draw(recon_nx, pos_recon, node_color=get_node_colors(recon_nx), 
            with_labels=True, node_size=300, font_size=8)
    plt.title(f"Reconstructed Graph\n({reconstructed_data['room'].num_nodes} rooms, {reconstructed_data['portal'].num_nodes} portals)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()



def evaluate_model_with_visualizations(model, test_loader, dataset, test_indices, device='cpu', num_samples=5):
    """
    Evaluate the model on the test set with visualizations.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        dataset: Original dataset
        test_indices: Indices of test samples in the dataset
        device: Device to run evaluation on
        num_samples: Number of samples to visualize
    """
    import os
    os.makedirs("reconstruction_visualizations", exist_ok=True)
    
    # Overall test metrics
    test_metrics = evaluate_reconstruction(model, test_loader, device=device)
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Sample-wise evaluation with visualizations
    print("\nRunning detailed evaluation on test samples:")
    samples_processed = 0
    
    for i, data in enumerate(test_loader):
        if samples_processed >= num_samples:
            break
            
        # If data is a batch, we need to handle it accordingly
        if isinstance(data, list):
            continue
            
        try:
            # Get original data for comparison
            test_idx = test_indices[i] if hasattr(test_loader.dataset, 'indices') else i
            original_data = dataset[test_idx]
            
            # Create masked version for reconstruction
            masked_data = remove_room_nodes(copy.deepcopy(data), remove_ratio=0.3)
            
            # Skip if masking removed all rooms
            if masked_data['room'].num_nodes == 0:
                continue
                
            # Run inference with statistics
            reconstructed_data, stats = run_inference_with_stats(
                model, masked_data, original_data=original_data, device=device)
            
            # Print statistics
            print(f"\nSample {samples_processed + 1}:")
            print_reconstruction_statistics(stats)
            
            # Visualize comparison
            save_path = f"reconstruction_visualizations/sample_{samples_processed}.png"
            visualize_reconstruction_comparison(
                original_data, masked_data, reconstructed_data, save_path=save_path)
            
            samples_processed += 1
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            continue
    
    print(f"\nProcessed {samples_processed} test samples with visualizations.")
    print("Visualizations saved in the 'reconstruction_visualizations' directory.")



def analyze_node_generation(model, data, device='cpu'):
    """
    Analyze the node generation process to understand what factors influence it.
    
    Args:
        model: Trained reconstruction model
        data: Input data with missing nodes
        device: Device to run analysis on
        
    Returns:
        Analysis results
    """
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        # Get graph-level embedding and node-level embeddings
        x_dict = {node_type: x for node_type, x in data.x_dict.items()}
        edge_index_dict = data.edge_index_dict
        
        # Forward pass to get embeddings
        _, _, _, missing_nodes_estimate, h_dict, graph_embedding = model(
            x_dict, edge_index_dict)
        
        # Number of missing nodes estimated
        num_missing = int(missing_nodes_estimate.item())
        
        # Generate embeddings for missing nodes
        generated_embeddings = model.generate_missing_nodes(
            h_dict, graph_embedding, torch.tensor([num_missing], device=device))
        
        # Predict node types for generated embeddings
        if generated_embeddings.size(0) > 0:
            generated_types = model.node_classifier(generated_embeddings).argmax(dim=1)
        else:
            generated_types = torch.tensor([], device=device)
        
        # Calculate attention weights for graph pooling
        attention_weights = {}
        for node_type, x in h_dict.items():
            attention_weights[node_type] = model.graph_pooling.attention(x).sigmoid()
        
        analysis = {
            'graph_embedding': graph_embedding.cpu().numpy(),
            'missing_nodes_estimate': num_missing,
            'generated_embeddings': generated_embeddings.cpu().numpy() if generated_embeddings.size(0) > 0 else None,
            'generated_types': generated_types.cpu().numpy() if generated_types.size(0) > 0 else None,
            'attention_weights': {k: v.cpu().numpy() for k, v in attention_weights.items()},
            'original_node_types': {k: model.node_classifier(v).argmax(dim=1).cpu().numpy() for k, v in h_dict.items() 
                                   if k == 'room'}
        }
        
        return analysis
    
    

def plot_training_curves(train_metrics, val_metrics):
    """
    Plot training and validation metrics over epochs.
    Args:
        train_metrics: Dictionary containing training metrics over epochs
        val_metrics: Dictionary containing validation metrics over epochs
    """
    plt.figure(figsize=(15, 10))
    
    # 1. Plot Losses
    plt.subplot(2, 2, 1)
    if len(train_metrics['epoch']) > 0 and len(train_metrics['loss']) > 0:
        plt.plot(train_metrics['epoch'], train_metrics['loss'], 'b-', label='Training Loss')
    if len(val_metrics['epoch']) > 0 and len(val_metrics['loss']) > 0:
        plt.plot(val_metrics['epoch'], val_metrics['loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 2. Plot Component Losses
    plt.subplot(2, 2, 2)
    if len(train_metrics['epoch']) > 0:  # Only check epochs as we'll use the same x-values for all lines
        if 'node_loss' in train_metrics and len(train_metrics['node_loss']) == len(train_metrics['epoch']):
            plt.plot(train_metrics['epoch'], train_metrics['node_loss'], 'r-', label='Node Loss')
        if 'link_loss' in train_metrics and len(train_metrics['link_loss']) == len(train_metrics['epoch']):
            plt.plot(train_metrics['epoch'], train_metrics['link_loss'], 'g-', label='Link Loss')
        if 'weight_loss' in train_metrics and len(train_metrics['weight_loss']) == len(train_metrics['epoch']):
            plt.plot(train_metrics['epoch'], train_metrics['weight_loss'], 'b-', label='Weight Loss')
        if 'existence_loss' in train_metrics and len(train_metrics['existence_loss']) == len(train_metrics['epoch']):
            plt.plot(train_metrics['epoch'], train_metrics['existence_loss'], 'c-', label='Existence Loss')
    plt.title('Training Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 3. Plot Node Metrics
    plt.subplot(2, 2, 3)
    if 'node_acc' in train_metrics and len(train_metrics['node_acc']) > 0:
        if len(train_metrics['node_acc']) != len(train_metrics['epoch']):
            # If lengths don't match, we need to create a properly aligned x-axis
            # Assuming validation metrics were collected at fixed intervals
            val_epochs = train_metrics['epoch'][:len(train_metrics['node_acc'])]
            plt.plot(val_epochs, train_metrics['node_acc'], 'm-', label='Train Node Acc')
        else:
            plt.plot(train_metrics['epoch'], train_metrics['node_acc'], 'm-', label='Train Node Acc')
            
    if 'node_f1' in val_metrics and len(val_metrics['node_f1']) > 0:
        if len(val_metrics['node_f1']) != len(val_metrics['epoch']):
            # Similar handling for validation metrics
            val_epochs = val_metrics['epoch'][:len(val_metrics['node_f1'])]
            plt.plot(val_epochs, val_metrics['node_f1'], 'y-', label='Val Node F1')
        else:
            plt.plot(val_metrics['epoch'], val_metrics['node_f1'], 'y-', label='Val Node F1')
    plt.title('Node Classification Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    
    # 4. Plot Reconstruction Metrics
    plt.subplot(2, 2, 4)
    if 'reconstruction_acc' in val_metrics and len(val_metrics['reconstruction_acc']) > 0:
        if len(val_metrics['reconstruction_acc']) != len(val_metrics['epoch']):
            val_epochs = val_metrics['epoch'][:len(val_metrics['reconstruction_acc'])]
            plt.plot(val_epochs, val_metrics['reconstruction_acc'], 'g-', label='Recon Accuracy')
        else:
            plt.plot(val_metrics['epoch'], val_metrics['reconstruction_acc'], 'g-', label='Recon Accuracy')
            
    if 'existence_mae' in val_metrics and len(val_metrics['existence_mae']) > 0:
        if len(val_metrics['existence_mae']) != len(val_metrics['epoch']):
            val_epochs = val_metrics['epoch'][:len(val_metrics['existence_mae'])]
            plt.plot(val_epochs, val_metrics['existence_mae'], 'k-', label='Existence MAE')
        else:
            plt.plot(val_metrics['epoch'], val_metrics['existence_mae'], 'k-', label='Existence MAE')
    plt.title('Validation Reconstruction Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    
    # Ensure output directory exists
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    filename = f"training_curves_{train_metrics['epoch'][-1]}.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Training curves saved as '{filepath}'")


def main_train(device, metadata):

    # Define reconstruction model
    model = ReconstructionHGTModel( hidden_channels=128,
                                    out_channels=len(ROOM_TYPE_MAP.keys()),
                                    num_heads=8,
                                    num_layers=3,
                                    metadata=metadata)

    # Train model
    model = train_reconstruction_model_with_monitoring(
        model, 
        train_loader,
        train_masked_loader,
        val_loader, 
        dataset,
        val_indices,
        epochs=5,  # Set higher epochs but rely on early stopping
        lr=0.001, 
        device=device,
        patience=10  # Stop after 10 epochs of no improvement
    )

    """     
    # Evaluate with visualizations
    evaluate_model_with_visualizations(
        model, 
        test_loader, 
        dataset, 
        test_indices, 
        device=device
    )        
    """
        
        
def main_test(device, metadata, size):
    
    # Load the model for inference
    model_path = 'best_reconstruction_model.pt'
    model = load_model_for_inference(model_path, metadata, device)
    
    # Evaluate on test set
    test_metrics = evaluate_reconstruction(model, test_loader, device=device)
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    if size > len(test_metrics):
        size = len(test_metrics)
    
    # Demonstrate reconstruction on test data AKA Run inference on test data
    print("\nRunning inference on test data:")
    for i, data in enumerate(test_loader):
        if i >= size:  # Just do a few examples
            break
            
        # If data is a batch, we need to handle it accordingly
        if isinstance(data, list):
            continue
            
        try:
            # Get original data for comparison
            test_idx = test_indices[i] if hasattr(test_loader.dataset, 'indices') else i
            original_data = dataset[test_idx]
            
            # Create masked version for reconstruction
            masked_data = remove_room_nodes(copy.deepcopy(data), remove_ratio=0.3)
            
            # Skip if masking removed all rooms
            if masked_data['room'].num_nodes == 0:
                continue
                
            # Run inference with statistics
            reconstructed_data, stats = run_inference_with_stats(model, masked_data, original_data=original_data, device=device)
            
            # Print statistics
            print_reconstruction_statistics(stats)
            
            # Visualize results
            # Assuming you have a dataset where each item is a tuple of (original_data, masked_data, reconstructed_data)
            visualize_masked_dataset_with_reconstructed(data, masked_data, reconstructed_data)
            
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            continue