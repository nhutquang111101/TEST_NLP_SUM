import os

import re
from bs4 import BeautifulSoup

import nltk
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# phục vụ xây dựng đồ thị và tính toán độ tương đồng
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from scipy.spatial import distance
import json

#vẽ đồ thị ra màn hình
import networkx as nx
import matplotlib.pyplot as plt




# STEP 1: đọc dữ liệu của Thư mục và File 
# Đường dẫn đến thư mục chứa các file

# Đường dẫn tệp output.txt nơi bạn muốn lưu toàn bộ dữ liệu đã đọc
output_file_path = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/output.txt'
# Đường dẫn tệp output.txt nơi bạn muốn lưu toàn bộ dữ liệu đã đọc
output_file_path_SUM = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/outputSUM.txt'

# Mở tệp output.txt để ghi dữ liệu
with open(output_file_path, 'w', encoding='utf-8') as output_file:
        with open(r'D:/WORKSPACE/CodeWork/nlpcode/myenv/train/d061j', 'r', encoding='utf-8') as file:
                content = file.read()
                output_file.write(content)  # Ghi nội dung tệp vào output.txt    

print(f"Dữ liệu đã được lưu vào {output_file_path}")

# STEP 2 đọc lại dữ liệu đã tạo và xử lý nó: Loại bỏ các thể của dữ liệu và tiền xử lý văn bản, như tách câu tách từ.
fp = open(output_file_path, 'r')
# read file: tao bien de luu lai gia tri
out_put_data = fp.read() 
# print(fp.read())
# Closing the file after reading
fp.close()

# tiến hành loại bỏ các thẻ S để lấy thành các câu

# Bước 1: Loại bỏ thẻ XML bằng BeautifulSoup
soup = BeautifulSoup(out_put_data, 'html.parser')
text = soup.get_text() 


# Tiền xử lý văn bản: Loại bỏ ký tự đặc biệt, chuyển thành chữ thường và loại bỏ dừng từ

# tách câu 
sentences = sent_tokenize(text)

# Kiểm tra và in ra các câu đã tách
# print("Danh sách các câu sau khi tách:")
# for idx, sentence in enumerate(sentences, 1):
#     print(f"Câu {idx}: {sentence}")



stop_words = set(stopwords.words('english'))

def preprocess_text(sentence):
    # Chuyển thành chữ thường
    sentence = sentence.lower()
    # Loại bỏ ký tự đặc biệt
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Tokenize và loại bỏ Hư từ
    tokens = word_tokenize(sentence)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# Áp dụng tiền xử lý cho từng câu
processed_sentences = [preprocess_text(sentence) for sentence in sentences]

# In kết quả sau khi tiền xử lý
# print("\nDanh sách các câu sau khi tiền xử lý:")
# for idx, sentence in enumerate(processed_sentences, 1):
#     print(f"Câu {idx}: {sentence}")


# Lưu vào file 
preprocessed_output_file = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/preprocessed_sentences.txt'
with open(preprocessed_output_file, 'w', encoding='utf-8') as file:
    for sentence in processed_sentences:
        file.write(sentence + '\n')

# END tiền xử lý

# Tính Toán để tính PageRank

# -- Tính toán TF IDF
contentTF = ''
with open(preprocessed_output_file, 'r', encoding='utf-8') as file:
    contentTF = file.read().splitlines()
       

# Tính toán độ tương đồng dữ liệu   
  
### 2. Xây dựng đồ thị dựa trên TF-IDF và độ tương đồng cosine
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_sentences)
feature_names = vectorizer.get_feature_names_out()  # Lấy danh sách từ vựng

# Lưu TF-IDF vào file
TFIDFFile = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/TF-IDF.txt'
with open(TFIDFFile, 'w', encoding='utf-8') as file:
    for doc_idx, row in enumerate(tfidf_matrix):  # Duyệt qua từng tài liệu
        file.write(f"Document {doc_idx + 1}:\n")
        row_data = row.toarray()[0]  # Chuyển thành array để dễ thao tác
        for word_idx, score in enumerate(row_data):
            if score > 0:  # Chỉ lưu những từ có trọng số > 0
                file.write(f"  {feature_names[word_idx]}: {score:.4f}\n")
        file.write("\n")

# Tính toán độ tương đồng cosine giữa các câu
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

cosine_sim_file = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/cosine_similarity_matrix.txt'

# Lưu ma trận độ tương đồng cosine vào file
np.savetxt(cosine_sim_file, cosine_sim_matrix, fmt='%.4f', delimiter='\t', header="Cosine Similarity Matrix")

# Xây dựng đồ thị từ ma trận độ tương đồng

# Đặt ngưỡng (threshold) để quyết định kết nối các node
threshold = 0.5 # Giá trị ngưỡng (chỉ kết nối nếu độ tương đồng lớn hơn ngưỡng này)

# Lọc ma trận cosine similarity
filtered_cosine_sim_matrix = np.where(cosine_sim_matrix >= threshold, cosine_sim_matrix, 0)

cosine_sim_matrix = filtered_cosine_sim_matrix
# Sử dụng từ điển để biểu diễn đồ thị
graph = {}

# Số lượng đỉnh trong đồ thị
num_nodes = cosine_sim_matrix.shape[0]

# Duyệt qua ma trận để xây dựng đồ thị
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j and cosine_sim_matrix[i, j] > 0:  #kiểm tra ngưỡng
            if i not in graph:
                graph[i] = []  # Nếu chưa có đỉnh i, khởi tạo danh sách rỗng
            graph[i].append((j, cosine_sim_matrix[i, j]))

# Đường dẫn lưu đồ thị
graph_file = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/graph_edge_list.txt'

# Lưu đồ thị dưới dạng danh sách cạnh
with open(graph_file, 'w', encoding='utf-8') as f:
    for node, edges in graph.items():
        for edge in edges:
            f.write(f"{node}\t{edge[0]}\t{edge[1]:.4f}\n")  # Ghi mỗi cạnh trên một dòng

# Đường dẫn lưu đồ thị dạng JSON
graph_json_file = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/graph.json'

# Chuyển đồ thị sang định dạng JSON
graph_data = {"nodes": [], "edges": []}

# Thêm thông tin các đỉnh
for node in graph.keys():
    graph_data["nodes"].append({"id": node})

# Thêm thông tin các cạnh
for node, edges in graph.items():
    for edge in edges:
        graph_data["edges"].append({
            "source": node,
            "target": edge[0],
            "weight": edge[1]
        })

# Lưu đồ thị dưới dạng JSON
with open(graph_json_file, 'w', encoding='utf-8') as f:
    json.dump(graph_data, f, ensure_ascii=False, indent=4)

# Chuyển đổi từ từ điển sang đồ thị NetworkX
G = nx.DiGraph()

# Thêm các cạnh vào đồ thị NetworkX
for i, neighbors in graph.items():
    for j, weight in neighbors:
        G.add_edge(i, j, weight=weight)

# Vẽ đồ thị
pos = nx.spring_layout(G)  # Bố cục đồ thị
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=12)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')

plt.title("Cosine Similarity Graph (Without Threshold)")
plt.show()

# Tính Pagerank
def pagerank(matrix, d=0.85, tol=1.0e-6, max_iter=100):
    # """
    # Tính điểm PageRank từ ma .
    
    N = matrix.shape[0]
    
    # Chuẩn hóa ma trận kề để tạo ma trận xác suất chuyển đổi
    degr = np.sum(matrix, axis=1)
    matrix_transaction = matrix / degr[:, None]
    matrix_transaction = np.nan_to_num(matrix_transaction)  # Xử lý chia cho 0
    
    # Khởi tạo điểm PageRank ban đầu (chia đều cho các nút)
    pagerank = np.ones(N) / N

    # Lặp tính toán điểm PageRank
    for iteration in range(max_iter):
        # Tính toán PageRank bằng công thức vector hóa
        new_pagerank = (1 - d) / N + d * np.dot(matrix_transaction.T, pagerank)
        
        # Kiểm tra hội tụ: nếu sự thay đổi nhỏ hơn ngưỡng tol thì dừng
        if np.linalg.norm(new_pagerank - pagerank, ord=1) < tol:
            break
        
        # Cập nhật giá trị PageRank
        pagerank = new_pagerank

    return pagerank
print(filtered_cosine_sim_matrix)
# Áp dụng thuật toán PageRank
pagerank_scores = pagerank(filtered_cosine_sim_matrix)

# Sắp xếp các câu theo điểm PageRank
ranked_sentences = sorted(((pagerank_scores[i], s) for i, s in enumerate(processed_sentences)), reverse=True)

# In ra điểm PageRank của từng câu
Score_pageRank = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/PageRank_Score.txt'
with open(Score_pageRank, 'w', encoding='utf-8') as file:
    for rank, sentence in ranked_sentences:
        file.write(f"Điểm: {rank:.4f} | Câu: {sentence}" + '\n')


# Tính số lượng câu trong văn bản gốc
total_sentences = len(ranked_sentences)

# Chọn số lượng câu tóm tắt theo tỷ lệ 10 %
summary_ratio = 0.1
num_sentences_summary = max(1, int(total_sentences * summary_ratio))  # Đảm bảo ít nhất 1 câu

# Lấy các câu tóm tắt từ danh sách xếp hạng
summary_sentences = [ranked_sentences[i][1] for i in range(num_sentences_summary)]


# Lưu vào file 
Sum_auto_path = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/SUM_AUTO.txt'
with open(Sum_auto_path, 'w', encoding='utf-8') as file:
    for sentence in summary_sentences:
        file.write(sentence + '\n')
print(f"Dữ liệu đã được lưu vào {Sum_auto_path}")


# ***** xử lý Văn bản đã tóm tắt và So sanh với văn bản đã tóm tắt
# Mở tệp output.txt để ghi dữ liệu
with open(output_file_path_SUM, 'w', encoding='utf-8') as output_file:
        with open(r'D:/WORKSPACE/CodeWork/nlpcode/myenv/DUC_SUM/d061j', 'r', encoding='utf-8') as file:
                content = file.read()
                output_file.write(content)  # Ghi nội dung tệp vào output.txt    

print(f"Dữ liệu đã được lưu vào {output_file_path_SUM}")

fp = open(output_file_path_SUM, 'r')
# read file: tao bien de luu lai gia tri
out_put_data = fp.read() 
# print(fp.read())
# Closing the file after reading
fp.close()

# tiến hành loại bỏ các thẻ S để lấy thành các câu

# Bước 1: Loại bỏ thẻ XML bằng BeautifulSoup
soup = BeautifulSoup(out_put_data, 'html.parser')
text = soup.get_text() 


# Tiền xử lý văn bản: Loại bỏ ký tự đặc biệt, chuyển thành chữ thường và loại bỏ dừng từ

# tách câu 
sentences = sent_tokenize(text)


# Áp dụng tiền xử lý cho từng câu
processed_sentences_SUM = [preprocess_text(sentence) for sentence in sentences]

# Lưu vào file 
preprocessed_output_file_SUM = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/preprocessed_sentences_SUM.txt'
with open(preprocessed_output_file_SUM, 'w', encoding='utf-8') as file:
    for sentence in processed_sentences_SUM:
        file.write(sentence + '\n')
        
contentTF_SUM = ''
with open(preprocessed_output_file_SUM, 'r', encoding='utf-8') as file:
    contentTF_SUM = file.read().splitlines()
    

# code mới 
# Chuyển danh sách thành tập hợp (set)
set1 = set(summary_sentences)  # Tập câu tóm tắt tự động
set2 = set(contentTF_SUM)  # Tập câu tóm tắt chuẩn

# So sánh các câu
common_sentences = set1 & set2  # Giao của hai tập hợp
common_count = len(common_sentences)  # Số lượng câu giống nhau

# Precision
precision = common_count / len(set1) if len(set1) > 0 else 0

# Recall
recall = common_count / len(set2) if len(set2) > 0 else 0

# F1-Score
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# In kết quả
print("Các câu giống nhau:")
print("\n".join(common_sentences))

print(f"\nSố câu giống nhau: {common_count}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")