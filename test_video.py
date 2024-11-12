import cv2
import numpy as np
import tensorflow as tf

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model('moblienetV3_200_clean.keras')

# Đọc các lớp (class names)
with open("test_anh/classes.names", "r", encoding="utf-8") as f:
    class_names = f.read().strip().split('\n')

# Đọc video đầu vào
cap = cv2.VideoCapture('test_anh/Chèo bẻo xám - Ashy Drongo bird.mp4')  # Thay 'video_input.mp4' bằng đường dẫn video của bạn

# Kiểm tra xem video có mở thành công không
if not cap.isOpened():
    print("Không thể mở video")
    exit()

while True:
    # Đọc một frame từ video
    ret, frame = cap.read()
    if not ret:
        break  # Nếu không còn frame nào, thoát khỏi vòng lặp

    # Tiền xử lý hình ảnh (resize và chuẩn hóa)
    input_frame = cv2.resize(frame, (224, 224))  # Resize theo kích thước đầu vào của mô hình
    input_frame = np.expand_dims(input_frame, axis=0)  # Thêm một chiều batch
    input_frame = input_frame.astype(np.float32) / 255.0  # Chuẩn hóa pixel [0, 1]

    # Dự đoán với mô hình
    predictions = model.predict(input_frame)

    # In ra predictions để kiểm tra cấu trúc dữ liệu


    # Giả sử predictions chứa xác suất của các lớp, chúng ta lấy lớp có xác suất cao nhất
    predicted_class = np.argmax(predictions[0])  # Lớp có xác suất cao nhất
    predicted_confidence = np.max(predictions[0])  # Xác suất của lớp đó

    # Kiểm tra nếu confidence > 0.5
    if predicted_confidence > 0.5:
        class_name = class_names[predicted_class]  # Tên lớp từ class_names

        # Vẽ tên lớp và xác suất lên frame
        cv2.putText(frame, f'{class_name}: {predicted_confidence:.2f}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị kết quả
    cv2.imshow('Object Classification', frame)

    # Nhấn 'q' để thoát khỏi video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
