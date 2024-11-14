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



# Vẽ đồ thị
import matplotlib.pyplot as plt



# STEP 1: đọc dữ liệu của Thư mục và File 
# Đường dẫn đến thư mục chứa các file
folder_path_train = '/Users/duyenhuynh/Documents/python/TEST_NLP_SUM/myenv/train'

# Đường dẫn tệp output.txt nơi bạn muốn lưu toàn bộ dữ liệu đã đọc
output_file_path = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/output.txt'

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
       
       
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(contentTF)

# Bước 2: Tạo ma trận tương đồng
similarity_matrix = cosine_similarity(tfidf_matrix)

# Bước 3: Xây dựng đồ thị và áp dụng PageRank
graph = nx.from_numpy_array(similarity_matrix)
scores = nx.pagerank(graph)

print(f"điểm: {graph}"+"\n")