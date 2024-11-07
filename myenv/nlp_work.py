import underthesea
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

# Bước 1: Tiền xử lý văn bản

# Tải stopwords tiếng Việt từ file
def load_vietnamese_stopwords():
    with open('stopwords/vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords_vn = f.read().splitlines()
    return stopwords_vn

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()  # Chuyển thành chữ thường
    text = text.translate(str.maketrans('', '', string.punctuation))  # Loại bỏ dấu câu
    return text

# Tiền xử lý tài liệu
def preprocess_document(document):
    sentences = re.split(r'[.!?]', document)  # Tách câu
    sentences = [sentence.strip() for sentence in sentences if sentence]  # Loại bỏ khoảng trắng thừa
    stopwords_vn = load_vietnamese_stopwords()  # Tải stopwords
    sentences = [' '.join([word for word in underthesea.word_tokenize(sentence) if word not in stopwords_vn]) for sentence in sentences]
    print("Câu Văn sau khi tiền xử lý: ", sentences)
    return sentences

# Bước 2: Xây dựng đồ thị
def build_similarity_matrix(sentences):
    tfidf_vectorizer = TfidfVectorizer()  # Khởi tạo TfidfVectorizer
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)  # Chuyển các câu thành ma trận TF-IDF
    cosine_similarities = (tfidf_matrix * tfidf_matrix.T).toarray()  # Tính ma trận tương đồng cosine giữa các câu
    print("xây dựng đồ thị: ", cosine_similarities)
    return cosine_similarities

# Bước 3: Áp dụng thuật toán PageRank
def apply_pagerank(similarity_matrix):
    G = nx.from_numpy_array(similarity_matrix)  # Tạo đồ thị từ ma trận tương đồng
    pagerank_scores = nx.pagerank(G)  # Tính PageRank cho các nút (câu)
    print("Tạo PageRank: ", pagerank_scores)
    return pagerank_scores

# Bước 4: Chọn các câu tóm tắt
def select_summary_sentences(pagerank_scores, sentences, top_n=3):
    ranked_sentences = sorted(((score, idx) for idx, score in pagerank_scores.items()), reverse=True)
    top_sentences = [sentences[idx] for _, idx in ranked_sentences[:top_n]]  # Lấy top n câu quan trọng
    print("Câu được chọn tóm tắt: ", top_sentences)
    return top_sentences

# Bước 5: Tổng hợp tóm tắt
def summarize_text(document, top_n=3):
    sentences = preprocess_document(document)  # Tiền xử lý và tách câu
    similarity_matrix = build_similarity_matrix(sentences)  # Xây dựng ma trận tương đồng
    pagerank_scores = apply_pagerank(similarity_matrix)  # Áp dụng thuật toán PageRank
    summary_sentences = select_summary_sentences(pagerank_scores, sentences, top_n)  # Chọn các câu quan trọng
    return ' '.join(summary_sentences)  # Kết hợp các câu thành một đoạn tóm tắt

# Ví dụ sử dụng
if __name__ == "__main__":
    document = """Ai đã từng nghe những giai điệu tha thiết, ngọt ngào trong khúc ca “Nhật kí của mẹ” chắc không thể nào quên “Này con yêu ơi, con biết không? Mẹ yêu con, yêu con rất nhiều! Những kỷ niệm lần đầu yêu, suốt một đời đâu dễ quên…” Người mẹ nào cũng yêu con rất nhiều, yêu con vô bờ bến. Mỗi lần nghe những ca từ về mẹ, trong em lại hiện về hình ảnh người mẹ thân thương của mình. Mẹ em năm nay vừa tròn bốn mươi, độ tuổi chưa phải quá già nhưng trên gương mặt đã hằn in bao vết tích mệt nhọc của thời gian. Mẹ có chiều cao khiêm tốn, nhưng vẻ nhỏ nhắn giúp thân hình mẹ cân đối. Mái tóc dài ngang vai, đen láy, suôn mượt càng tôn lên làn da trắng trẻo của mẹ. Làn da ấy đã có vài nếp nhăn cùng những chấm tàn nhang li ti trên gương mặt. Em hiểu những nếp nhăn đó là bao vất vả mẹ gánh chịu để nuôi dạy anh em em. Gương mặt mẹ trái xoan, xinh đẹp. Mẹ có đôi mắt nâu đen sâu thẳm. Người ta bảo, những người mắt sâu là những người sống tình cảm. Quả thực, mẹ em rất tình cảm, lúc nào trong mẹ cũng đầy ắp tình thương yêu cho gia đình, cho chúng em và cho những thế hệ học trò mà mẹ dạy dỗ. Đôi môi trái tim hồng hồng của mẹ lúc nào cũng nở nụ cười rạng rỡ. Chỉ khi anh em em chưa vâng lời, nụ cười đó mới vụt tắt. Khi đó, chiếc mũi cao của mẹ cũng đỏ hoe vì rơi nước mắt, trông mẹ buồn đến lạ. Em thích nhất chiếc má lúm đồng tiền của mẹ. Chiếc má lúm làm gương mặt mẹ thêm phần tươi xinh, duyên dáng. Mẹ em thường mặc quần vải sẫm màu cùng áo sơ mi dài tay. Trong bộ trang phục này, trông mẹ rất nghiêm túc nhưng không kém phần trẻ trung. Những ngày lễ, mẹ thướt tha trong tà áo dài. Em yêu nhất là đôi bàn tay mẹ, đôi bàn tay mềm mại nhưng đã có chút nhăn nheo. Làm sao mà không nhăn nheo khi đôi bàn tay ấy đã chăm bẵm từng miếng ăn, giấc ngủ cho anh em em, đã dắt dìu bao thế hệ học trò bỡ ngỡ vào lớp một? Mẹ em là một người vừa ấm áp, vừa nghiêm khắc. Ngày còn nhỏ, mẹ mua cho chúng tôi rất nhiều đồ chơi. Nhưng mẹ cũng căn dặn sau khi chơi xong phải cất ngăn nắp và gìn giữ cẩn thận. Bài hát “Nhật kí của mẹ” kể lại cả chặng đường người mẹ sinh thành, nuôi dưỡng và thương yêu người con nhưng có lẽ chẳng câu từ nào có thể đong đếm được sự hi sinh và tình cảm bao la mẹ dành cho con. Em luôn thầm hứa sẽ ngoan ngoãn, vâng lời cha mẹ và học tập thật tốt để không phụ lòng sinh dưỡng của mẹ."""
    
    summary = summarize_text(document, top_n=2)  # Tóm tắt văn bản và lấy 2 câu quan trọng nhất
    print("Tóm tắt:", summary)
