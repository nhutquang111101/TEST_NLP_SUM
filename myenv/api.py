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
import pandas as pd
from scipy.spatial import distance



# Vẽ đồ thị
import matplotlib.pyplot as plt



# STEP 1: đọc dữ liệu của Thư mục và File 
# Đường dẫn đến thư mục chứa các file
folder_path_train = '/Users/duyenhuynh/Documents/python/TEST_NLP_SUM/myenv/train'
folder_path_SUM = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/train/d061j'

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
       
# doc_len = len(contentTF)
# # print(doc_len)
       
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(contentTF)

# # Bước 2: Tạo ma trận tương đồng
# similarity_matrix = cosine_similarity(tfidf_matrix)
    
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(contentTF)

# # Bước 2: Tạo ma trận tương đồng
# similarity_matrix = cosine_similarity(tfidf_matrix)

# # Bước 3: Xây dựng đồ thị và áp dụng PageRank
# graph = nx.from_numpy_array(similarity_matrix)
# scores = nx.pagerank(graph)

# print(f"điểm: {graph}"+"\n")

# Tính toán độ tương đồng dữ liệu   
  
### 2. Xây dựng đồ thị dựa trên TF-IDF và độ tương đồng cosine
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_sentences)

# Tính toán độ tương đồng cosine giữa các câu
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# Xây dựng đồ thị từ ma trận độ tương đồng
graph = nx.from_numpy_array(cosine_sim_matrix)


# print(cosine_sim_matrix )

# Tính Pagerank
def pagerank(matrix, d=0.85, tol=1.0e-6, max_iter=100):
    # """
    # Tính điểm PageRank từ ma trận kề.
    
    # Args:
    # - matrix: Ma trận kề (numpy array).
    # - d: Hệ số giảm dần (default 0.85).
    # - tol: Ngưỡng hội tụ (default 1.0e-6).
    # - max_iter: Số lần lặp tối đa (default 100).
    
    # Returns:
    # - ranks: Điểm PageRank của từng nút.
    # """
    N = matrix.shape[0]
    
    # Chuẩn hóa ma trận kề để tạo ma trận xác suất chuyển đổi
    out_degree = np.sum(matrix, axis=1)
    transition_matrix = matrix / out_degree[:, None]
    transition_matrix = np.nan_to_num(transition_matrix)  # Xử lý chia cho 0
    
    # Khởi tạo điểm PageRank
    ranks = np.ones(N) / N
    
    for iteration in range(max_iter):
        new_ranks = (1 - d) / N + d * np.dot(transition_matrix.T, ranks)
        
        # Kiểm tra hội tụ
        if np.linalg.norm(new_ranks - ranks, 1) < tol:
            break
        ranks = new_ranks

    return ranks

# Áp dụng thuật toán PageRank
pagerank_scores = pagerank(cosine_sim_matrix)
# print("Điểm PageRank cho từng câu:")
# print(pagerank_scores)

# Sắp xếp các câu theo điểm PageRank
ranked_sentences = sorted(((pagerank_scores[i], s) for i, s in enumerate(processed_sentences)), reverse=True)

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

# sum_len = len(summary_sentences)
# print(sum_len)
# In các câu tóm tắt
# print("Tóm tắt văn bản final:")
# for sentence in summary_sentences:
#     print("-", sentence)
    
# ### 5. Tổng hợp tóm tắt
# summary = ' '.join(summary_sentences)
# print("Tóm tắt văn bản:")
# print(summary)

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
    
    
 ### 2. Xây dựng đồ thị dựa trên TF-IDF và độ tương đồng cosine
vectorizer_SUM = TfidfVectorizer()
tfidf_matrix_SUM = vectorizer_SUM.fit_transform(processed_sentences_SUM)

# # tính độ tương đồng của 2 văn bản
# # Ngưỡng Cosine Similarity để xác định câu tương đồng
# similarity_threshold = 0.7

# # Kết hợp cả 2 danh sách lại để tính toán TF-IDF
# all_summaries = summary_sentences + contentTF_SUM
# tfidf_matrix = vectorizer.fit_transform(all_summaries)
# # Kết hợp cả 2 danh sách lại để tính toán TF-IDF
# all_summaries = summary_sentences + contentTF_SUM  # Kết hợp hai danh sách
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(all_summaries)  # Tính toán TF-IDF cho toàn bộ câu

# # Tách riêng các vector của tóm tắt tự động và chuẩn
# auto_summary_vectors = tfidf_matrix[:len(summary_sentences)]  # Các vector của câu từ auto_summary
# reference_summary_vectors = tfidf_matrix[len(contentTF_SUM):]  # Các vector của câu từ reference_summary


# Chuyển danh sách thành tập hợp (set)
set1 = set(summary_sentences)
set2 = set(contentTF_SUM)

# So sánh các câu
common_sentences = set1 & set2  # Giao của hai tập hợp
common_count = len(common_sentences)  # Số lượng câu giống nhau

# In kết quả

print("Các câu giống nhau:")
print("\n".join(common_sentences))

print(f"\nSố câu giống nhau: {common_count}")
# Tính phần trăm
percent_dataset1 = (common_count / len(set1)) * 100
percent_dataset2 = (common_count / len(set2)) * 100
print(f"Phần trăm câu giống nhau trong dataset1: {percent_dataset1:.2f}%")
print(f"Phần trăm câu giống nhau trong dataset2: {percent_dataset2:.2f}%")