import re
import time
import os
import torch
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class LLM():
    def __init__(self, key: str | None = None, model_name: str | None = None):
        # API key và model cho Groq GPT-OSS
        self.key = key or os.getenv("GROQ_API_KEY")
        if not self.key:
            raise ValueError("Missing GROQ_API_KEY. Please set env or pass key explicitly.")
        self.model_name = model_name or os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

        # Khởi tạo Groq client
        self.client = Groq(api_key=self.key)

    def answer(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý AI trả lời ngắn gọn, chính xác bằng tiếng Việt."},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    
    def embed(self, query: str):
        """
        Trả về embedding của câu truy vấn.
        Lưu ý: Groq GPT-OSS hiện không cung cấp endpoint embedding tiêu chuẩn.
        Vui lòng tích hợp sentence-transformers hoặc dịch vụ embedding khác nếu cần.
        """
        raise NotImplementedError("Embedding is not implemented for Groq GPT-OSS in this project.")

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
    # emded = model.embed(query)
    # print(emded)
    enterties, relattioships = model.extract_entities_and_relationships(query)
    print(enterties)