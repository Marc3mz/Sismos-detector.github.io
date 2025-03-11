import os
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Descargar datos del USGS
def download_data():
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": "2023-09-01",
        "endtime": "2023-10-01",
        "minmagnitude": 1.0,
        "maxlatitude": 12.0,
        "minlatitude": -56.0,
        "maxlongitude": -34.0,
        "minlongitude": -82.0
    }
    response = requests.get(url, params=params)
    data = response.json()
    return pd.json_normalize(data['features'])

# Preprocesar datos
def preprocess_data(earthquakes):
    earthquakes['longitude'] = earthquakes['geometry.coordinates'].apply(lambda x: x[0])
    earthquakes['latitude'] = earthquakes['geometry.coordinates'].apply(lambda x: x[1])
    earthquakes['depth'] = earthquakes['geometry.coordinates'].apply(lambda x: x[2])
    features = ['properties.mag', 'depth', 'longitude', 'latitude']
    X = earthquakes[features]
    y = (earthquakes['properties.mag'] >= 5.0).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
def train_model(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = xgb.XGBClassifier(eval_metric='logloss')  # Eliminado use_label_encoder
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo: {accuracy:.2f}')
    return model

# Guardar modelo
def save_model(model, path):
    # Crear la carpeta si no existe
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)

# Ejecutar todo el proceso
if __name__ == "__main__":
    earthquakes = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(earthquakes)
    model = train_model(X_train, X_test, y_train, y_test)
    save_model(model, 'model/earthquake_model.json')