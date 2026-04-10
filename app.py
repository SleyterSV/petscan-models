import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # 🔥 IMPORTANTE: Importamos CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# 🔥 IMPORTANTE: Activamos CORS para que Flutter (Chrome/Móvil) pueda leer la respuesta sin bloqueos
CORS(app, resources={r"/*": {"origins": "*"}})

# 1. CARGAR TU MODELO .H5 🧠
MODEL_PATH = 'modelo_pet_disease.h5'
try:
    model = load_model(MODEL_PATH)
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    print(f"🔴 ERROR FATAL: No se pudo cargar el modelo. Verifica que el archivo exista. Detalle: {e}")

# ESTAS SON LAS CLASES QUE TU MODELO RECONOCE
# OJO: He puesto las enfermedades que vi en tu base de datos.
# Asegúrate de que el orden sea exactamente el mismo con el que entrenaste a la IA.
CLASS_NAMES = ['Alergia cutánea general', 'Sarna Sarcóptica', 'Granuloma por lamido', 'Sano'] 

def prepare_image(img_bytes):
    """Prepara la imagen para que el modelo la pueda entender."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # Ajusta este tamaño (224, 224) al tamaño exacto que usaste para entrenar tu modelo.
    img = img.resize((224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalización
    return img_array

@app.route('/', methods=['GET'])
def home():
    return "🐶 PetScanIA API está funcionando al 100%. El CORS está activado. Usa /predict para enviar imágenes."

# Agregamos 'OPTIONS' para que los navegadores web hagan el pre-chequeo de CORS sin problemas
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'OK'}), 200

    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen', 'success': False}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío', 'success': False}), 400

    try:
        # A. Preparar la imagen
        img_bytes = file.read()
        processed_image = prepare_image(img_bytes)

        # B. Hacer la predicción con tu modelo
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # C. Obtener el nombre del diagnóstico
        if predicted_class_index < len(CLASS_NAMES):
            diagnosis = CLASS_NAMES[predicted_class_index]
        else:
            diagnosis = f"Clase desconocida ({predicted_class_index})"

        print(f"🤖 IA Predijo: {diagnosis} con {confidence*100:.2f}% de certeza")

        # D. Devolver el resultado a la app (Flutter)
        # 🔥 Usamos las variables que Flutter espera para que no haya errores
        return jsonify({
            'diagnostico': diagnosis, 
            'confianza': confidence,
            'success': True
        })

    except Exception as e:
        print(f"🔴 Error en la predicción: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    # Render usa la variable de entorno PORT
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
