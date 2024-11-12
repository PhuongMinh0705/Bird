import cv2
import numpy as np
import tensorflow as tf
import time

# Tải mô hình đã lưu
model = tf.keras.models.load_model(r'moblienetV3_200_clean.keras')


# Hàm tiền xử lý ảnh
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Kích thước đầu vào của MobileNetV3
    frame = frame / 255.0  # Chuẩn hóa dữ liệu
    frame = np.expand_dims(frame, axis=0)  # Thêm chiều batch
    return frame


# Mở video
cap = cv2.VideoCapture('test_anh/minh.mp4')  # Đường dẫn video của bạn

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Tiền xử lý khung hình
    input_frame = preprocess_frame(frame)

    # Dự đoán
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions, axis=-1)

    # Hiển thị kết quả
    class_names = ['Bách thanh mày trắng', 'Bìm bịp lớn', 'Bói cá nhỏ', 'Bông lau mày trắng', 'Bạc má',
               'Bồ nông chân xám', 'Bồng chanh', 'Bồng chanh lam sáng', 'Chim Xanh Trán Vàng', 'Chim khách',
               'Chim manh', 'Chim sâu bụng vạch', 'Chim sâu lưng đỏ', 'Chim sâu mỏ lớn', 'Chiền chiện lớn',
               'Chiền chiện đầu nâu', 'Choi choi sông', 'Choắt lớn', 'Choắt nâu', 'Chèo bẻo', 'Chèo bẻo xám',
               'Chìa vôi trắng', 'Chích bông nâu', 'Chích chòe than', 'Chích đuôi dài', 'Cu cu đen', 'Cu gáy',
               'Cu ngói', 'Cu rốc cổ đỏ', 'Cu vằn', 'Cu xanh đầu xám', 'Cuốc ngực trắng', 'Cà kheo', 'Cò bợ',
               'Cò bợ Mã Lai', 'Cò lùn hung', 'Cò lùn nâu', 'Cò lửa lùn', 'Cò ngàng lớn', 'Cò nhạn',
               'Cò quăm đầu đen', 'Cò ruồi', 'Cò trắng', 'Cò xanh', 'Cốc đế', 'Cổ rắn', 'Di đá', 'Diều hoa Miến Điện',
               'Diều trắng', 'Diệc lửa', 'Diệc xám', 'Dô nách nâu', 'Giang sen', 'Già đẫy Java', 'Gà lôi nước',
               'Gà lôi nước Ấn Độ', 'Gõ kiến nhỏ nâu xám', 'Gõ kiến vàng lớn', 'Gõ kiến xanh bụng vàng',
               'Hút mật họng nâu', 'Hút mật đỏ', 'Hạc cổ trắng', 'Kịch', 'Le hôi', 'Le nâu', 'Mòng két mày trắng',
               'Nghệ ngực vàng', 'Nhạn bụng trắng', 'Nhạn rừng', 'Phướn', 'Phường chèo trắng lớn',
               'Quăm đầu đen', 'Quắm đen', 'Rẻ quạt java', 'Rồng rộc', 'Sáo nâu', 'Sáo sậu', 'Sáo vai trắng',
               'Sáo đá đuôi hung', 'Sả khoang cổ', 'Sả mỏ rộng', 'Sả rừng', 'Sả đầu nâu', 'Sả đầu đen',
               'Sẻ bụi vàng', 'Sẻ bụi đen', 'Sẻ nhà', 'Sếu sarus', 'Te vặt', 'Trích cồ', 'Trảu họng vàng',
               'Trảu ngực nâu', 'Trảu đầu hung', 'Tu hú', 'Tìm vịt', 'Vành khuyên họng vàng', 'Vạc',
               'Vẹt đầu hồng', 'Vịt mốc', 'Vịt trời', 'Yểng', 'Yểng quạ', 'Ó cá', 'Đuôi cụt cánh xanh',
               'Đớp ruồi nâu', 'Đớp ruồi xanh gáy đen', 'Đớp ruồi xanh xám']  # Danh sách tên các loài chim
    label = class_names[predicted_class[0]]

    # Vẽ nhãn lên khung hình
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Hiển thị khung hình
    cv2.imshow('Video', frame)

    # Delay để điều chỉnh tốc độ xử lý
    time.sleep(0.05)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
