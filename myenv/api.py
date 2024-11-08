import os

import re
from bs4 import BeautifulSoup

import nltk
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# phục vụ xây dựng đồ thị và tính toán độ tương đồng
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



# Vẽ đồ thị
import matplotlib.pyplot as plt



# STEP 1: đọc dữ liệu của Thư mục và File 
# Đường dẫn đến thư mục chứa các file
folder_path_train = '/Users/duyenhuynh/Documents/python/TEST_NLP_SUM/myenv/train'

# Đường dẫn tệp output.txt nơi bạn muốn lưu toàn bộ dữ liệu đã đọc
output_file_path = '/Users/duyenhuynh/Documents/python/TEST_NLP_SUM/myenv/output.txt'

# Mở tệp output.txt để ghi dữ liệu
with open(output_file_path, 'w', encoding='utf-8') as output_file:
        with open(r'/Users/duyenhuynh/Documents/python/TEST_NLP_SUM/myenv/train/d061j', 'r', encoding='utf-8') as file:
                content = file.read()
                output_file.write(content)  # Ghi nội dung tệp vào output.txt    

print(f"Dữ liệu đã được lưu vào {output_file_path}")

# STEP 2 đọc lại dữ liệu đã tạo và xử lý nó: Loại bỏ các thể của dữ liệu và tiền xử lý văn bản, như tách câu tách từ.
fp = open(r'/Users/duyenhuynh/Documents/python/TEST_NLP_SUM/myenv/output.txt', 'r')
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
    # Tokenize và loại bỏ từ dừng
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
# preprocessed_output_file = '/Users/duyenhuynh/Documents/python/TEST_NLP_SUM/myenv/preprocessed_sentences.txt'
# with open(preprocessed_output_file, 'w', encoding='utf-8') as file:
#     for sentence in processed_sentences:
#         file.write(sentence + '\n')



# Xây dựng đồ thị biểu diễn 


# Bước 1: Tính toán vector TF-IDF cho các câu
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_sentences) 

# Bước 2: Tính ma trận độ tương đồng cosine
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# print("Tính toán độ tương đồng: ",cosine_sim_matrix)

# Bước 3: Xây dựng đồ thị
graph = nx.Graph()


# Thêm các nút vào đồ thị (mỗi nút là một câu)
for i in range(len(processed_sentences)):
    graph.add_node(i, sentence=processed_sentences[i])

# Thêm các cạnh vào đồ thị dựa trên độ tương đồng (chỉ thêm nếu độ tương đồng > ngưỡng, ví dụ 0.1)
threshold = 0.1
for i in range(len(processed_sentences)):
    for j in range(i+1, len(processed_sentences)):
        if cosine_sim_matrix[i][j] > threshold:
            graph.add_edge(i, j, weight=cosine_sim_matrix[i][j])


# In thông tin đồ thị
# print("\nThông tin các cạnh trong đồ thị:")
# for edge in graph.edges(data=True):
#     print(f"Nút {edge[0]} liên kết với nút {edge[1]} với trọng số {edge[2]['weight']:.4f}")



# Bước 4: Hiển thị ma trận tương đồng dưới dạng bảng
print("\nMa trận độ tương đồng cosine:")
print(np.round(cosine_sim_matrix, 2))

# Bước 5: Vẽ đồ thị (Tùy chọn - yêu cầu thư viện matplotlib)

# plt.figure(figsize=(8, 6))
# pos = nx.spring_layout(graph, seed=42)  # Bố trí các nút
# nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue')
# nx.draw_networkx_edges(graph, pos, edgelist=graph.edges, width=1.0, edge_color='gray')
# nx.draw_networkx_labels(graph, pos, labels={i: f"Câu {i+1}" for i in range(len(processed_sentences))}, font_size=10)
# plt.title("Đồ thị biểu diễn các câu dựa trên độ tương đồng")
# plt.show()

# Lưu lại data vào File sau khi xử lý xong văn bản
# with open(output_file_path, 'w', encoding='utf-8') as output_file:
#     output_file.write(text)


def calculate_pagerank(similarity_matrix, alpha=0.85, max_iter=100, tol=1e-6):
    # Số lượng nút (câu)
    N = similarity_matrix.shape[0]

    # Khởi tạo điểm PageRank cho mỗi nút
    pagerank_scores = np.ones(N) / N

    # Chuẩn hóa ma trận độ tương đồng để lấy ma trận chuyển tiếp
    transition_matrix = similarity_matrix / similarity_matrix.sum(axis=1, keepdims=True)

    for iteration in range(max_iter):
        new_pagerank_scores = (1 - alpha) / N + alpha * np.dot(transition_matrix.T, pagerank_scores)
        
        # Kiểm tra điều kiện hội tụ
        if np.linalg.norm(new_pagerank_scores - pagerank_scores, ord=1) < tol:
            print(f"PageRank hội tụ sau {iteration+1} vòng lặp.")
            break
        
        pagerank_scores = new_pagerank_scores

    return pagerank_scores

# Tính điểm PageRank
pagerank_scores = calculate_pagerank(cosine_sim_matrix)

# Sắp xếp các câu theo điểm PageRank
sorted_indices = np.argsort(-pagerank_scores)
# print("\nXếp hạng các câu theo PageRank:")
# for idx, sentence_index in enumerate(sorted_indices, 1):
#     print(f"Hạng {idx}: Câu {sentence_index+1} (Điểm PageRank: {pagerank_scores[sentence_index]:.4f}) - {processed_sentences[sentence_index]}")


# Lấy tất cả các câu sau khi sắp xếp theo điểm PageRank
all_sentence_indices = sorted_indices

# Sắp xếp lại các câu theo thứ tự xuất hiện ban đầu trong văn bản
all_sentence_indices = sorted(all_sentence_indices)

# In tất cả các câu theo thứ tự ban đầu
# print("\nTóm tắt văn bản với tất cả các câu (sắp xếp theo PageRank):")
# for idx, sentence_index in enumerate(all_sentence_indices, 1):
#     print(f"Câu {idx}: {processed_sentences[sentence_index]}")

# Lấy tất cả các câu sau khi sắp xếp theo điểm PageRank
all_sentence_indices = sorted_indices

# Sắp xếp lại các câu theo thứ tự xuất hiện ban đầu trong văn bản
all_sentence_indices = sorted(all_sentence_indices)

# Tổng hợp các câu đã chọn thành một đoạn văn tóm tắt
summary = " ".join([processed_sentences[i] for i in all_sentence_indices])

# In kết quả tóm tắt
print("\nTóm tắt văn bản với tất cả các câu (sắp xếp theo PageRank):")
print(summary)