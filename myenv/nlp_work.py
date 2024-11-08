import os

import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# STEP 1: đọc dữ liệu của Thư mục và File 
# Đường dẫn đến thư mục chứa các file
folder_path_train = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/train'

# Đường dẫn tệp output.txt nơi bạn muốn lưu toàn bộ dữ liệu đã đọc
output_file_path = 'D:/WORKSPACE/CodeWork/nlpcode/myenv/output.txt'

# Mở tệp output.txt để ghi dữ liệu
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    # Lặp qua các file trong thư mục
    for filename in os.listdir(folder_path_train):
        file_path = os.path.join(folder_path_train, filename)
        if os.path.isfile(file_path):
            # output_file.write(f"Nội dung của file {filename}:\n")  # Ghi tiêu đề cho mỗi tệp
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                output_file.write(content)  # Ghi nội dung tệp vào output.txt
                # output_file.write("\n" + "-" * 50 + "\n")  # Ngăn cách giữa các file
        else:
            output_file.write(f"{filename} không phải là File mà là Thư mục\n")

print(f"Dữ liệu đã được lưu vào {output_file_path}")

# STEP 2 đọc lại dữ liệu đã tạo và xử lý nó


