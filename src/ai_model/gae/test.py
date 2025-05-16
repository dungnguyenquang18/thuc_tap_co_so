from neo4j import GraphDatabase
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rapidfuzz import process  # For fuzzy matching
from collections import deque
from thuc_tap_co_so.src.ai_model.gae import GAE
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
import re

# Load environment
load_dotenv()

# Cấu hình kết nối Neo4j
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)

# Cấu hình Gemini
genai.configure(api_key=os.getenv('API_KEY'))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


def extract_entities_and_relationships(text, model):
    prompt = (
        f"Trích xuất các thực thể (nút) và mối quan hệ (cạnh) từ văn bản dưới đây."
        f"Các thực thể và mối quan hệ PHẢI được viết bằng tiếng Việt.\n"
        f"Tuân theo định dạng sau:\n\n"
        f"Thực thể:\n"
        f"- {{Thực thể}}: {{Loại}}\n\n"
        f"Mối quan hệ:\n"
        f"- ({{Thực thể1}}, {{Loại quan hệ}}, {{Thực thể2}})\n\n"
        f"Văn bản:\n\"{text}\"\n\n"
        f"Đầu ra:\nThực thể:\n- {{Thực thể}}: {{Loại}}\n...\n\n"
        f"Mối quan hệ:\n- ({{Thực thể1}}, {{Loại quan hệ}}, {{Thực thể2}})\n"
    )

    for _ in range(5):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Lỗi: {e}")
            time.sleep(10)

    return ""


def process_llm_out(response_text):
    entity_pattern = r"- (.+): (.+)"
    entities = re.findall(entity_pattern, response_text)
    entity_dict = {entity.strip(): entity_type.strip() for entity, entity_type in entities}
    entity_list = list(entity_dict.keys())

    relationship_pattern = r"- \(([^,]+), ([^,]+), ([^)]+)\)"
    relationships = re.findall(relationship_pattern, response_text)
    relationship_list = [
        (s.strip(), r.strip().replace(" ", "_").upper(), o.strip())
        for s, r, o in relationships
    ]

    return entity_list, relationship_list


def get_graph_data():
    with driver.session() as session:
        node_records = session.run("MATCH (n:Entity) RETURN id(n) AS node_id, n.name AS name")
        node_list = [(record["node_id"], record["name"]) for record in node_records]
        node_mapping = {original_id: idx for idx, (original_id, _) in enumerate(node_list)}
        node_names = {idx: name for idx, (_, name) in enumerate(node_list)}

        edge_records = session.run("MATCH (n)-[r]->(m) RETURN id(n) AS source, id(m) AS target, type(r) AS relationship_type")
        edge_list = [
            (node_mapping[record["source"]], node_mapping[record["target"]], record["relationship_type"])
            for record in edge_records
        ]

        unique_relationship_types = {e[2] for e in edge_list}
        return node_names, edge_list, unique_relationship_types


def find_indirect_connection(edge_list, node_mapping, start, target, max_depth=10):
    graph = {}
    for src, tgt, rel in edge_list:
        graph.setdefault(src, []).append((tgt, rel))
        graph.setdefault(tgt, []).append((src, rel))

    queue = deque([(start, [], 0)])
    visited = set()
    paths = []

    while queue:
        current_node, path, depth = queue.popleft()
        if depth > max_depth:
            continue
        if current_node == target:
            paths.append(path)
            continue
        visited.add(current_node)
        for neighbor, relationship in graph.get(current_node, []):
            if neighbor not in visited:
                queue.append((neighbor, path + [(current_node, relationship, neighbor)], depth + 1))
    return paths


def find_closest_entities(entities, node_mapping):
    results = []
    node_names = list(node_mapping.values())
    for entity in entities:
        closest_match, score, index = process.extractOne(entity, node_names)
        closest_match_id = list(node_mapping.keys())[index]
        results.append((entity, closest_match_id, closest_match, score))
    return results


def retrieve_information(query, k=20):
    node_mapping, edge_list, unique_relationship_types = get_graph_data()
    edge_index = torch.tensor([[e[0], e[1]] for e in edge_list], dtype=torch.long).t().contiguous()

    num_nodes = len(node_mapping)
    features = torch.eye(num_nodes)

    graph_data = Data(x=features, edge_index=edge_index)

    input_dim = features.size(1)
    hidden_dim = 16
    embedding_dim = 8

    model = GAE(input_dim, hidden_dim, embedding_dim)

    if not os.path.exists('gae.torch'):
        raise FileNotFoundError("Model file 'gae.torch' not found. Please train and save the GAE model.")

    model.load_state_dict(torch.load('gae.torch'))
    model.eval()

    with torch.no_grad():
        embeddings, _ = model(graph_data.x, graph_data.edge_index)

    query_output = extract_entities_and_relationships(query, gemini_model)
    entities, _ = process_llm_out(query_output)

    matches = find_closest_entities(entities, node_mapping)

    node_embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    information = ''

    for query_entity, match_id, match_name, score in matches:
        query_embedding = F.normalize(embeddings[match_id], p=2, dim=0)
        similarity_scores = torch.matmul(query_embedding.unsqueeze(0), node_embeddings_norm.T).squeeze()
        top_k_indices = torch.topk(similarity_scores, k).indices

        for idx in top_k_indices:
            similar_node_id = idx.item()
            similar_node_name = node_mapping[similar_node_id]
            similarity_score = similarity_scores[idx].item()

            direct_connections = [
                e for e in edge_list
                if (e[0] == match_id and e[1] == similar_node_id) or
                   (e[1] == match_id and e[0] == similar_node_id)
            ]

            if direct_connections:
                for connection in direct_connections:
                    source = node_mapping[connection[0]]
                    target = node_mapping[connection[1]]
                    relationship = connection[2]
                    information += f"{source} -> {relationship} -> {target}.\n"
            else:
                paths = find_indirect_connection(edge_list, node_mapping, match_id, similar_node_id)
                if paths:
                    for path in paths:
                        formatted_path = " -> ".join(
                            f"{node_mapping[src]} -[{rel}]-> {node_mapping[tgt]}" for src, rel, tgt in path
                        )
                        information += formatted_path + '.\n'
                else:
                    information += f"{match_name} -> NO_DIRECT_RELATION -> {similar_node_name}.\n"
    return information


if __name__ == '__main__':
    query = 'hãy nêu điều 2 mục 3 luật đất đai?'
    result = retrieve_information(query)
    print(result)
