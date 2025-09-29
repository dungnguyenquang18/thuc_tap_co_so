from flask import Flask, request, jsonify
from thuc_tap_co_so.src.ai_model import Retrieve
from thuc_tap_co_so.src.ai_model import LLM
import dotenv
import os
import random
dotenv.load_dotenv()


app = Flask(__name__)
KEYS = [k for k in [os.getenv('GROQ_API_KEY'), os.getenv('GROQ_API_KEY_2'), os.getenv('GROQ_API_KEY_3')] if k]


@app.route('/api/chatbot', methods=['POST'])
def handle_query():
    if not KEYS:
        return jsonify({'error': 'No API keys configured'}), 500
    retrieve = Retrieve(random.choice(KEYS))
    llm = LLM()
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    information = retrieve.retrieve_infomation(query)

    answer = llm.answer(f"hãy trả lời câu hỏi sau:({query}) từ dữ liệu sau:\n{information}")
    return jsonify({"answer": answer})


if __name__ == '__main__':
    app.run(debug=True, port=5001)