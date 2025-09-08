🩺 Dự đoán Khối U bằng Random Forest + Flask + Bootstrap 5
📌 Giới thiệu

Ứng dụng web dự đoán khối u ác tính hay lành tính dựa trên dữ liệu xét nghiệm.

Thuật toán: Random Forest Classifier (Scikit-learn)

Giao diện web: Flask + Bootstrap 5

Dữ liệu: Breast Cancer Wisconsin (Diagnostic) Dataset

📂 Cấu trúc thư mục
📦 Project
 ┣ 📜 app.py                  # Flask web app (gọi model để dự đoán)
 ┣ 📜 train_model.py          # Script train & lưu model Random Forest
 ┣ 📜 random_forest_model.pkl # File mô hình đã train (sinh ra sau khi chạy train_model.py)
 ┣ 📜 data.csv                # Dataset (dùng để train)
 ┣ 📂 templates
 ┃ ┗ 📜 index.html            # Giao diện web (Bootstrap 5)
 ┗ 📜 README.md               # Tài liệu hướng dẫn

 <img width="1912" height="853" alt="image" src="https://github.com/user-attachments/assets/7e1a5c42-6a17-497f-ac48-09742314e71a" />
