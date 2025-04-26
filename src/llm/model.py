import re
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv

load_dotenv()


class LLM():
    def __init__(self):
        # Thay thế bằng API Key của bạn
        self.API_KEY = os.getenv('API_KEY')
        # Cấu hình API Key
        genai.configure(api_key=self.API_KEY)
        # Khởi tạo mô hình Gemini Pro
        self.model = genai.GenerativeModel("gemini-1.5-pro")

    def answer(self, query):
        result = self.model.generate_content(query).text
        return result

    def extract_entities_and_relationships(self, text):

        prompt = (
            f"Extract entities (nodes) and their relationships (edges) from the text below."
            f"Entities and relationships MUST be in Vietnamese\n"
            f"Follow this format:\n\n"
            f"Entities:\n"
            f"- {{Entity}}: {{Type}}\n\n"
            f"Relationships:\n"
            f"- ({{Entity1}}, {{RelationshipType}}, {{Entity2}})\n\n"
            f"Text:\n\"{text}\"\n\n"
            f"Output:\nEntities:\n- {{Entity}}: {{Type}}\n...\n\n"
            f"Relationships:\n- ({{Entity1}}, {{RelationshipType}}, {{Entity2}})\n"
        )

        for _ in range(5):  # Gửi 5 request
            try:
                response = self.answer(prompt)
                return self.process_llm_out(response)
            except Exception as e:
                print(f"Lỗi: {e}")
                time.sleep(20)  # Chờ 10 giây trước khi thử lại

        return 'lỗi api'

    def process_llm_out(self, response):

        # Extract entities
        entity_pattern = r"- (.+): (.+)"
        entities = re.findall(entity_pattern, response)
        entity_dict = {entity.strip(): entity_type.strip() for entity, entity_type in entities}
        # { "Samsung Galaxy A100": "Sản phẩm", "Pin": "Thành phần"}
        entity_list = list(entity_dict.keys())

        # Extract relationships
        relationship_pattern = r"- \(([^,]+), ([^,]+), ([^)]+)\)"
        relationships = re.findall(relationship_pattern, response)
        relationship_list = [(subject.strip(), relation.strip().replace(" ", "_").upper(), object_.strip()) for
                             subject, relation, object_ in relationships]

        # Output entities and relationships
        print("Entities:")
        for entity, entity_type in entity_dict.items():
            print(f"{entity}: {entity_type}")

        print("\nRelationships:")
        for subject, relation, object_ in relationship_list:
            print(f"({subject}, {relation}, {object_})")

        return entity_list, relationship_list


if __name__ == '__main__':
    model = LLM()
    query = 'hãy nêu điều 2 khoản 1 luật đất đai'
    enterties, relattioships = model.extract_entities_and_relationships(query)
    print(enterties)