from flask import Flask, request, jsonify
from thuc_tap_co_so.src.ai_model.retrieve import Retrieve
from thuc_tap_co_so.src.ai_model.llm import LLM

app = Flask(__name__)
retrieve = Retrieve()
llm = LLM()


@app.route('/api/chatbot', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    information = retrieve.retrieve_infomation(query)

    answer = llm.answer(f"hãy trả lời câu hỏi sau:({query}) từ dữ liệu sau:\n{information}")
    return jsonify({"answer": answer})


if __name__ == '__main__':
    app.run(debug=True, port=5001)