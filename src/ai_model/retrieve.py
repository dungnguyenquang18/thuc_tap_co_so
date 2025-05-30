from neo4j import GraphDatabase
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rapidfuzz import process  # For fuzzy matching
from collections import deque
from thuc_tap_co_so.src.ai_model.gae import GAE
import os
from dotenv import load_dotenv
from thuc_tap_co_so.src.ai_model.llm import LLM

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
        edge_list = [(node_mapping[record["source"]], node_mapping[record["target"]], record["relationship_type"]) for
                     record in edges]

        # Lấy danh sách các loại quan hệ duy nhất
        unique_relationship_types = {e[2] for e in edge_list}

        return node_names, edge_list, unique_relationship_types


def find_indirect_connection(edge_list, node_mapping, start, target, max_depth=10):
    # Build graph as adjacency list: {node: [(neighbor, relationship)]}
    graph = {}
    for src, tgt, rel in edge_list:
        if src not in graph:
            graph[src] = []
        if tgt not in graph:
            graph[tgt] = []
        graph[src].append((tgt, rel))
        graph[tgt].append((src, rel))  # Assuming undirected graph for traversal

    # Initialize BFS
    queue = deque([(start, [], 0)])  # (current_node, path_so_far, current_depth)
    visited = set()

    paths = []

    while queue:
        current_node, path, depth = queue.popleft()

        if depth > max_depth:  # Stop exploring if max depth is exceeded
            continue

        if current_node == target:  # Target node found
            paths.append(path)
            continue

        # Mark node as visited
        visited.add(current_node)

        # Explore neighbors
        for neighbor, relationship in graph.get(current_node, []):
            if neighbor not in visited:
                queue.append((neighbor, path + [(current_node, relationship, neighbor)], depth + 1))

    return paths


def find_closest_entities(entities, node_mapping):
    """
    Finds the closest matching entities in node_mapping for a list of query entities.

    Parameters:
        entities (list): List of entity names to match.
        node_mapping (dict): Mapping of node IDs to entity names.

    Returns:
        list: A list of tuples [(query_entity, closest_match_id, closest_match_name, score)].
    """
    results = []
    node_names = list(node_mapping.values())
    for entity in entities:
        closest_match, score, index = process.extractOne(entity, node_names)
        closest_match_id = list(node_mapping.keys())[index]
        results.append((entity, closest_match_id, closest_match, score))
    return results




class Retrieve():
    def __init__(self):
        pass

    def retrieve_infomation(self, query, k=5):
        node_mapping, edge_list, unique_relationship_types = get_graph_data()
        edge_index = torch.tensor([[e[0], e[1]] for e in edge_list], dtype=torch.long).t().contiguous()

        num_nodes = len(node_mapping)
        features = torch.eye(num_nodes)

        graph_data = Data(x=features, edge_index=edge_index)

        input_dim = features.size(1)
        hidden_dim = 16
        embedding_dim = 8

        model = GAE(input_dim, hidden_dim, embedding_dim)

        # if not os.path.exists('gae2.torch'):
        #     raise FileNotFoundError("Model file 'gae.torch' not found. Please train and save the GAE model.")

        model.load_state_dict(torch.load('D:/3Y2S/ttcs2/thuc_tap_co_so/src/ai_model/gae2.torch'))
        model.eval()

        with torch.no_grad():
            embeddings, _ = model(graph_data.x, graph_data.edge_index)

        llm = LLM()
        entities, _ = llm.extract_entities_and_relationships(query)

        matches = find_closest_entities(entities, node_mapping)

        node_embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        information = ''

        for query_entity, match_id, match_name, score in matches:
            print(f"\nTop-{k} similar nodes for '{query_entity}' (Matched Node: {match_name}):")
            query_embedding = embeddings[match_id]
            query_embedding = F.normalize(query_embedding, p=2, dim=0)

            # Compute similarity scores
            similarity_scores = torch.matmul(query_embedding.unsqueeze(0), node_embeddings_norm.T).squeeze()

            # Retrieve Top-K similar nodes
            top_k_indices = torch.topk(similarity_scores, k).indices

            for idx in top_k_indices:
                similar_node_id = idx.item()
                similarity_score = similarity_scores[idx].item()
                similar_node_name = node_mapping[similar_node_id]

                # Check for direct connection in the edge list
                direct_connections = [
                    e for e in edge_list if (e[0] == match_id and e[1] == similar_node_id) or
                                            (e[1] == match_id and e[0] == similar_node_id)
                ]

                # Print details in the desired format
                if direct_connections:
                    for connection in direct_connections:
                        source = node_mapping[connection[0]]
                        target = node_mapping[connection[1]]
                        relationship = connection[2]
                        information += f"{source} -> {relationship} -> {target}.\n"
                        print(f"{source} -> {relationship} -> {target} with score {similarity_score:.4f}")
                else:
                    # Print indirect connection paths
                    paths = find_indirect_connection(edge_list, node_mapping, match_id, similar_node_id)
                    if paths:
                        print(f"Indirect paths between '{match_name}' and '{similar_node_name}':")
                        for path in paths:
                            formatted_path = " -> ".join(
                                f"{node_mapping[src]} -[{rel}]-> {node_mapping[tgt]}" for src, rel, tgt in path
                            )
                            information += formatted_path + '.\n'
                            print(formatted_path)
                    else:
                        information += f"{match_name} -> NO_DIRECT_RELATION -> {similar_node_name}.\n"
                        print(
                            f"{match_name} -> NO_DIRECT_RELATION -> {similar_node_name} with score {similarity_score:.4f}")

        return information


if __name__ == '__main__':
    retrieve = Retrieve()
    query = 'cho tôi biết về hành vi vi phạm quy định về đấu thầu?'
    info = retrieve.retrieve_infomation(query)
    print(info)