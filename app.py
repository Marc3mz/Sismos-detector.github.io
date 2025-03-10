from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo entrenado
model_path = 'model/earthquake_model.json'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"El archivo {model_path} no existe.")

model = xgb.XGBClassifier()
model.load_model(model_path)

# Ruta principal (interfaz web)
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del frontend
        data = request.get_json()
        mag = float(data['mag'])
        depth = float(data['depth'])
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])

        # Crear un array con los datos para la predicción
        features = np.array([[mag, depth, longitude, latitude]])

        # Hacer la predicción
        prediction = model.predict(features)

        # Devolver la predicción como JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return jsonify({'error': str(e)}), 500

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)