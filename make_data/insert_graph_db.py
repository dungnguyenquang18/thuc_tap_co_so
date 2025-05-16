from pymongo import MongoClient
#connect to db
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
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
# print(NEO4J_URI)
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
        
    def insert_by_using_mongodb_content(self):
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

    def insert_by_using_mongodb_paper(self):
        try:
            uri = os.getenv('URI')
            client = MongoClient(uri)
            db = client[os.getenv('DB_NAME')]
            collection = db[os.getenv('COLLECTION_NAME')]
        except:
            print('lỗi do chưa chạy db')

        data = list(collection.find())
        for i,paper in enumerate(data):
            content = paper['content']
            title = paper['title']
            input = 'tiêu đề bài báo: ' + title + '\nnội dung:\n ' + content
            api_key = [os.getenv('API_KEY_1'),os.getenv('API_KEY_2'),os.getenv('API_KEY_3')]
            result = extract_entities_and_relationships(input, api_key[i%3])
            _, relationship_list = process_llm_out(result)
            add_relationships_to_neo4j(self.kg, relationship_list)
                            
    def insert_by_using_local_file(self):
        pass
            
        


def extract_entities_and_relationships(text, API_KEY):
    # text = sample

    prompt = (
        f"Trích xuất các thực thể (nút) và mối quan hệ (cạnh) từ các baài báo luật pháp dưới đây."
        f"Các thực thể và mối quan hệ PHẢI được viết bằng tiếng Việt.\n"
        f"Tuân theo định dạng sau:\n\n"
        f"Thực thể:\n"
        f"- {{Thực thể}}: {{Loại}}\n\n"
        f"Mối quan hệ:\n"
        f"- ({{Thực thể1}}, {{Loại quan hệ}}, {{Thực thể2}})\n\n"
        f"Văn bản:\n\"{text}\"\n\n"
        f"Đầu ra:\nThực thể:\n- {{Thực thể}}: {{Loại}}\n...\n\n"
        f"Mối quan hệ:\n- ({{Thực thể1}}, {{Loại quan hệ}}, {{Thực thể2}})\n"
        f"các thực thể và các mối quan phải được tạo ra bằng việc đặt câu hỏi liên quan đến luật pháp:\n"
        f"1. Về nguồn và tính pháp lý"
        f"Bài báo có dẫn nguồn chính thức hay văn bản pháp luật cụ thể không?\n"
        f"Đây là ý kiến bình luận của cá nhân, cơ quan báo chí, hay trích dẫn từ cơ quan nhà nước có thẩm quyền?\n"
        f"Văn bản pháp luật được đề cập có còn hiệu lực không (có bị thay thế, sửa đổi chưa)?\n"
        f"2. Về nội dung pháp lý\n"
        f"Quy định pháp luật nào đang được nói đến (luật, nghị định, thông tư...)?\n"
        f"Quy định đó áp dụng trong trường hợp nào, đối tượng nào?\n"
        f"Có điều khoản chuyển tiếp hoặc ngoại lệ nào không?\n"
        f"Có án lệ hoặc vụ việc cụ thể nào làm tiền lệ được nhắc tới không?\n"
        f"3. Về đối tượng và hoàn cảnh áp dụng\n"
        f"Quy định này ảnh hưởng đến ai? (cá nhân, doanh nghiệp, tổ chức...)\n"
        f"Trong hoàn cảnh nào quy định này được kích hoạt hay thực thi?\n"
        f"Có yếu tố tranh chấp, kiện tụng hay xử phạt nào không?\n"
        f"4. Về quyền và nghĩa vụ\n"
        f"Quyền lợi của các bên liên quan được bảo đảm như thế nào?\n"
        f"Có nghĩa vụ nào phát sinh từ quy định này không?\n"
        f"Có chế tài gì nếu vi phạm?\n"
        f"6. Về tác động thực tiễn:\n"
        f"Quy định này sẽ ảnh hưởng như thế nào đến đời sống xã hội hoặc hoạt động kinh doanh?\n"
        f"Có phản ứng hoặc tranh luận gì từ xã hội, giới luật sư, chuyên gia không?"
    )





    # Thay thế bằng API Key của bạn


    # Cấu hình API Key
    genai.configure(api_key=API_KEY)

    # Khởi tạo mô hình Gemini Pro
    model = genai.GenerativeModel("gemini-2.0-flash")
    for _ in range(5):  # Gửi 5 request
        try:
            response = model.generate_content(prompt)
            return (response.text)
        except Exception as e:
            print(f"Lỗi: {e}")
            time.sleep(20)  # Chờ 10 giây trước khi thử lại
    

    # In kết quả
    return response.text

import re
from unidecode import unidecode

def sanitize_relation(relation):
    # Bỏ dấu tiếng Việt
    relation = unidecode(relation)
    # Đổi sang chữ in hoa
    relation = relation.upper()
    # Thay các ký tự không phải chữ/số bằng dấu gạch dưới
    relation = re.sub(r'[^A-Z0-9]', '_', relation)
    # Bỏ các dấu gạch dưới thừa
    relation = re.sub(r'_+', '_', relation).strip('_')
    return relation

def process_llm_out(result):
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
    relationship_list = []
    for subject, relation, object_ in relationships:
        subject = subject.strip()
        relation_clean = sanitize_relation(relation.strip())
        object_ = object_.strip()
        relationship_list.append((subject, relation_clean, object_))

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
    insert.insert_by_using_mongodb_paper()