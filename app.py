import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# 1. CARGAR TU MODELO .H5 🧠
# Asegúrate de que 'modelo_pet_disease.h5' esté en la raíz de tu repositorio junto a este archivo.
MODEL_PATH = 'modelo_pet_disease.h5'
model = load_model(MODEL_PATH)
print("✅ Modelo cargado correctamente.")

# ESTAS SON LAS CLASES QUE TU MODELO RECONOCE
# ¡IMPORTANTE! Reemplázalas con las enfermedades reales que tu IA sabe detectar, en el orden correcto.
CLASS_NAMES = ['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3'] 

def prepare_image(img_bytes):
    """Prepara la imagen para que el modelo la pueda entender."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # Ajusta este tamaño (224, 224) al tamaño exacto que usaste para entrenar tu modelo.
    img = img.resize((224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalización (si tu modelo la usó)
    return img_array

@app.route('/', methods=['GET'])
def home():
    return "🐶 PetScanIA API está funcionando. Usa /predict para enviar imágenes."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    try:
        # A. Preparar la imagen
        img_bytes = file.read()
        processed_image = prepare_image(img_bytes)

        # B. Hacer la predicción con tu modelo
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        diagnosis = CLASS_NAMES[predicted_class_index]

        # C. Devolver el resultado a la app
        return jsonify({
            'diagnosis': diagnosis,
            'confidence': confidence,
            'message': f"Diagnóstico IA: {diagnosis} (Confianza: {confidence:.2f})"
        })

    except Exception as e:
        print(f"🔴 Error en la predicción: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Render usa la variable de entorno PORT
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
