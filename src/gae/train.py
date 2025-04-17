from neo4j import GraphDatabase
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import random
from rapidfuzz import process  # For fuzzy matching
from collections import deque
from graph_rag_for_review_film_chatbot.src.gae.model import GAE
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()


driver = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))
def get_graph_data():
    with driver.session() as session:
        # Lấy danh sách các node
        nodes_query = "MATCH (n:Entity) RETURN id(n) AS node_id, n.name AS name"
        nodes = session.run(nodes_query)
        
        # Chuyển đổi thành danh sách để đảm bảo có thể đánh số lại
        node_list = [(record["node_id"], record["name"]) for record in nodes]
        
        # Tạo ánh xạ từ id gốc sang id liên tục bắt đầu từ 0
        node_mapping = {original_id: idx for idx, (original_id, _) in enumerate(node_list)}
        node_names = {idx: name for idx, (_, name) in enumerate(node_list)}

        # Lấy danh sách các cạnh và ánh xạ ID node theo node_mapping mới
        edges_query = "MATCH (n)-[r]->(m) RETURN id(n) AS source, id(m) AS target, type(r) AS relationship_type"
        edges = session.run(edges_query)
        edge_list = [(node_mapping[record["source"]], node_mapping[record["target"]], record["relationship_type"]) for record in edges]

        # Lấy danh sách các loại quan hệ duy nhất
        unique_relationship_types = {e[2] for e in edge_list}

        return node_names, edge_list, unique_relationship_types



def loss_function(reconstructed, edge_index):
        # Binary cross-entropy loss for adjacency reconstruction
        # Một tensor toàn giá trị 1 với chiều dài bằng số lượng edges (M) trong đồ thị.
        target = torch.ones(edge_index.size(1))  # All edges exist
        # Sự khác biệt giữa logits dự đoán (reconstructed) và các giá trị thực (target).
        return F.binary_cross_entropy_with_logits(reconstructed, target)




if __name__ == '__main__':
    node_mapping, edge_list, unique_relationship_types = get_graph_data()
    # Create a mapping for edge names to indices
    edge_name_to_index = {name: idx for idx, name in enumerate(set(edge[2] for edge in edge_list))}

    # Convert edge list to indices
    edge_index = [(src, tgt, edge_name_to_index[name]) for src, tgt, name in edge_list]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    reverse_node_mapping = {value: key for key, value in node_mapping.items()}
        
        
    # Step 1: Create edge_index
    edge_index = torch.tensor([[e[0], e[1]] for e in edge_list], dtype=torch.long).t().contiguous()

    # Features for nodes (one-hot encoding)
    num_nodes = len(node_mapping)
    features = torch.eye(num_nodes)  # One-hot encoding for each node

    # Split edges into training and validation sets
    num_edges = edge_index.size(1)
    indices = list(range(num_edges))
    random.shuffle(indices)
    split_idx = int(0.8 * num_edges)  # 80% for training, 20% for validation

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]

    # Step 2: Create Data objects for training and validation
    train_data = Data(x=features, edge_index=train_edge_index)
    val_data = Data(x=features, edge_index=val_edge_index)

    target = torch.ones(edge_index.size(1))
    
    # Step 4: Initialize the model
    input_dim = features.size(1)
    hidden_dim = 16
    embedding_dim = 8
    model = GAE(input_dim, hidden_dim, embedding_dim)

    # Step 5: Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    

    # Step 6: Train the model and validate after each epoch
    model.train()
    for epoch in range(200):
        # Training phase
        optimizer.zero_grad()
        embeddings, reconstructed = model(train_data.x, train_data.edge_index)
        train_loss = loss_function(reconstructed, train_data.edge_index)
        train_loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_embeddings, val_reconstructed = model(val_data.x, val_data.edge_index)
            val_loss = loss_function(val_reconstructed, val_data.edge_index)

        # Switch back to train mode for next epoch
        model.train()

        # Print losses
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}')

    # Lưu trạng thái mô hình
    torch.save(model.state_dict(), "gae.torch")
    print("Mô hình đã được lưu!")

