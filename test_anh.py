import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog, messagebox

# Load the model
model = tf.keras.models.load_model('moblienetV3_200_clean.keras')

# Define class names
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
               'Đớp ruồi nâu', 'Đớp ruồi xanh gáy đen', 'Đớp ruồi xanh xám']

def predict_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]

        return f"Image: {image_path} - Predicted label: {predicted_label}"
    except Exception as e:
        return f"Error processing {image_path}: {e}"

def select_images():
    Tk().withdraw()
    file_paths = filedialog.askopenfilenames(title="Select Images for Prediction",
                                             filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    results = []
    for file_path in file_paths:
        result = predict_image(file_path)
        results.append(result)

    # Show results in a message box
    messagebox.showinfo("Prediction Results", "\n".join(results))

    # Save results to a text file
    with open('frame.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(results))

def main():
    select_images()

if __name__ == "__main__":
    main()
