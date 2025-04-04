from pymongo import MongoClient
#connect to db
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os
import google.generativeai as genai
import time

# Tải biến môi trường từ file .env
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

class InsertIntoGraph():
    def __init__(self):

        try: 
            self.kg = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE
            )
        except:
            print('lỗi các biến trong môi trường hoặc chưa chạy db')
        
    def insert_by_using_mongodb(self):
        #set up collection
        try:
            uri = os.getenv('URI')
            client = MongoClient(uri)
            db = client[os.getenv('DB_NAME')]
            collection = db[os.getenv('COLLECTION_NAME')]
        except:
            print('lỗi do chưa chạy db')
        
        #load to dict
        data = list(collection.find())

        for law in data:
            content = law['noi_dung']
            thong_tin_ban_hanh = law['ten_luat'] + ' được ban hành bởi ' + law['co_quan_ban_hanh'] + '. ' + law['luat_so'] + '. Được ban hành vào ' + law['ngay_ban_hanh']
            for chuong, dieu in content.items():
                thong_tin_chuong = thong_tin_ban_hanh + '. ' + chuong

                for dieu_thu, noi_dung_dieu in dieu.items():
                    thong_tin_dieu = thong_tin_chuong + '. ' + dieu_thu + '. ' 

                    for khoan in (noi_dung_dieu):
                        thong_tin_khoan = thong_tin_dieu + ': ' + khoan 

                        result = extract_entities_and_relationships(thong_tin_khoan)
                        _, relationship_list = process_llm_out(result)
                        add_relationships_to_neo4j(self.kg, relationship_list)
        
                            
    def insert_by_using_local_file(self):
        pass
            
        


def extract_entities_and_relationships(text):
    # text = sample

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





    # Thay thế bằng API Key của bạn
    API_KEY = os.getenv('API_KEY')

    # Cấu hình API Key
    genai.configure(api_key=API_KEY)

    # Khởi tạo mô hình Gemini Pro
    model = genai.GenerativeModel("gemini-2.0-pro")
    for _ in range(5):  # Gửi 5 request
        try:
            response = model.generate_content(prompt)
            return (response.text)
        except Exception as e:
            print(f"Lỗi: {e}")
            time.sleep(20)  # Chờ 10 giây trước khi thử lại
    

    # In kết quả
    return response.text

def process_llm_out( result):
    import re

    # OpenAI response as a string
    response = result

    # Extract entities
    entity_pattern = r"- (.+): (.+)"
    entities = re.findall(entity_pattern, response)
    entity_dict = {entity.strip(): entity_type.strip() for entity, entity_type in entities}
    # { "Samsung Galaxy A100": "Sản phẩm", "Pin": "Thành phần"}
    entity_list = list(entity_dict.keys())

    # Extract relationships
    relationship_pattern = r"- \(([^,]+), ([^,]+), ([^)]+)\)"
    relationships = re.findall(relationship_pattern, response)
    relationship_list = [(subject.strip(), relation.strip().replace(" ", "_").upper(), object_.strip()) for subject, relation, object_ in relationships]

    # Output entities and relationships
    print("Entities:")
    for entity, entity_type in entity_dict.items():
        print(f"{entity}: {entity_type}")

    print("\nRelationships:")
    for subject, relation, object_ in relationship_list:
        print(f"({subject}, {relation}, {object_})")
    return entity_list, relationship_list

def add_relationships_to_neo4j(kg, relationships):
    """
    Add relationships extracted from the knowledge graph to the Neo4j database.
    """
    with kg._driver.session() as session:
        for subject, relation, obj in relationships:
            # Create nodes and relationships in Neo4j

            cypher_query = f"""
            MERGE (a:Entity {{name: $subject}})
            MERGE (b:Entity {{name: $object}})
            MERGE (a)-[:`{relation}`]->(b)
            """
            session.run(cypher_query, subject=subject, object=obj)
    print("Relationships added to Neo4j.")



if __name__ == '__main__':
    insert = InsertIntoGraph()
    insert.insert_by_using_local_file()