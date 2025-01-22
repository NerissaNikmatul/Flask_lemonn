from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Muat model yang telah dilatih
model = joblib.load('lemon_classifier.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/qualify', methods=['POST'])
def qualify():
    try:
        # Ambil file gambar dari formulir
        file = request.files['lemon_image']

        # Pastikan file diunggah
        if not file:
            return jsonify({"error": "No file uploaded"})

        # Buka gambar dan ubah ukurannya sesuai dengan input model
        image = Image.open(file)
        image = image.resize((224, 224))  # Sesuaikan ukuran dengan kebutuhan model
        image_array = np.array(image) / 255.0  # Normalisasi pixel
        image_array = image_array.reshape(1, 224, 224, 3)  # Bentuk input model

        # Lakukan prediksi
        prediction = model.predict(image_array)[0]
        result = "Baik" if prediction > 0.5 else "Buruk"

        # Tampilkan hasil prediksi di halaman yang sama
        return render_template('index.html', qualification=f"Hasil prediksi menunjukkan bahwa kualitas lemon adalah: {result}")
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)