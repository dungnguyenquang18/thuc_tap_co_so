import graphviz

def create_dataflow_diagram(output_path):
    dot = graphviz.Digraph(format='png')
    dot.attr(rankdir='LR', size='10,6')

    # Các node
    dot.node('User', 'Người dùng', shape='oval', style='filled', fillcolor='#e3f2fd')
    dot.node('Web', 'Giao diện web (Streamlit)', shape='box', style='filled', fillcolor='#fff9c4')
    dot.node('API', 'API backend (Flask)', shape='box', style='filled', fillcolor='#ffe0b2')
    dot.node('LLM1', 'LLM1\n(Trích xuất thực thể/quan hệ)', shape='box', style='filled', fillcolor='#c8e6c9')
    dot.node('GAE', 'GAE\n(Graph AutoEncoder)', shape='box', style='filled', fillcolor='#b3e5fc')
    dot.node('LLM2', 'LLM2\n(Sinh câu trả lời)', shape='box', style='filled', fillcolor='#d1c4e9')
    dot.node('Neo4j', 'Neo4j', shape='cylinder', style='filled', fillcolor='#b2dfdb')

    # Các mũi tên luồng dữ liệu
    dot.edge('User', 'Web', label='Nhập câu hỏi')
    dot.edge('Web', 'API', label='Gửi câu hỏi')
    dot.edge('API', 'LLM1', label='Chuyển truy vấn')
    dot.edge('LLM1', 'GAE', label='Truy xuất embedding/quan hệ')
    dot.edge('GAE', 'Neo4j', label='Truy vấn đồ thị', dir='both')
    dot.edge('GAE', 'LLM2', label='Kết quả embedding/tri thức')
    dot.edge('LLM1', 'LLM2', label='Thực thể/quan hệ trích xuất', style='dashed')
    dot.edge('LLM2', 'API', label='Trả câu trả lời')
    dot.edge('API', 'Web', label='Trả kết quả')
    dot.edge('Web', 'User', label='Hiển thị trả lời')

    dot.render(output_path, cleanup=True)

if __name__ == '__main__':
    create_dataflow_diagram('D:/3Y2S/ttcs2/thuc_tap_co_so/dataflow') 