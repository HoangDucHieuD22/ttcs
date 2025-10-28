# TTCS – Dự án phân vùng và phân loại tế bào máu  
(Translated: “TTCS – Blood Cell Segmentation & Classification Project”)

## 🧬 Giới thiệu  
Dự án TTCS được phát triển bởi **Hoàng Đức Hiếu** trong khuôn khổ đồ án cá nhân về xử lý ảnh y tế.  
Mục tiêu chính: sử dụng mô hình **CNN kết hợp với StarDist** để thực hiện phân vùng và phân loại tế bào bạch cầu trong ảnh hiển vi, phục vụ nghiên cứu và ứng dụng trong lĩnh vực y-sinh.

## 💡 Công nghệ & công cụ sử dụng  
- Ngôn ngữ: Python  
- Framework: StarDist (phục hồi và mở rộng mô hình CNN)  
- Thư viện hỗ trợ: NumPy, SciPy, scikit-image, TensorFlow/Keras (hoặc PyTorch nếu áp dụng)  
- Quá trình xử lý: Tiền xử lý ảnh → phân vùng (segmentation) → trích xuất đặc trưng → phân loại  
- Hệ thống quản lý mã nguồn: Git  
- Môi trường phát triển: Jupyter Notebook / VS Code

## 🎯 Vai trò của cá nhân  
- Tiền xử lý và làm sạch dữ liệu ảnh hiển vi  
- Xây dựng và huấn luyện mô hình StarDist để phân vùng tế bào  
- Triển khai pipeline inference để dự đoán mới trên dữ liệu thực  
- Đánh giá kết quả: đo lường độ chính xác, độ nhạy, F1-score giữa các lớp tế bào  

## 🗂 Cấu trúc thư mục 
/TTCS
|— data/ # ảnh gốc và ảnh đã tiền xử lý
|— notebooks/ # Jupyter notebooks thực nghiệm
|— models/ # mô hình đã huấn luyện (.h5, .pt…)
|— src/ # code chính (Python scripts)
|— results/ # kết quả chạy thử và báo cáo
|— README.md 
|— .gitignore 

bash
Copy code

## 🚀 Hướng dẫn chạy thử  
1. Clone repo về máy:  
   ```bash
   git clone https://github.com/HoangDucHieuD22/ttcs.git
   cd ttcs
Cài đặt môi trường (ví dụ bằng venv hoặc conda):

bash
Copy code
python -m venv env
source env/bin/activate   # trên Linux/Mac
env\Scripts\activate      # trên Windows
pip install -r requirements.txt
Chạy notebook hoặc script chính:

bash
Copy code
jupyter notebook notebooks/Segmentation_Classification.ipynb
Hoặc

bash
Copy code
python src/run_inference.py --input data/test_image.png --output results/output.png
Xem kết quả tại thư mục results/.

📊 Kết quả & suy nghĩ
Mô hình đạt được độ chính xác x%, mô hình phân vùng đúng được y% tế bào bạch cầu.

Rút ra bài học: cần tăng dữ liệu huấn luyện, cải thiện augmentation và cân nhắc mô hình phức tạp hơn để phân biệt các loại tế bào khó.
