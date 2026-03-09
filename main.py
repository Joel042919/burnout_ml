from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import xgboost as xgb

app = FastAPI(title="Burnout Prediction API")

# Configurar CORS para permitir que tu frontend se comunique con la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción en Render, puedes cambiar "*" por la URL de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo guardado (pipeline completo)
# Asegúrate de que 'imbalanced-learn' esté en tu requirements.txt para que esto no falle en Render
model_pipeline = joblib.load("burnout_model.pkl")

# Definir la estructura de los datos que enviará el frontend
class BurnoutInput(BaseModel):
    Gender: int
    Age_code: int = Field(alias="Age code")
    BMI: int
    Designation: int
    Specialization: int
    Working_Place: int
    Duration: int
    Private_Practice: int
    Working_Hour: int
    Weekly_Vacation: int
    Marital_Status: int
    Income: int
    Family_members: int
    Liabilities: int = Field(alias="Liabilities ") # Mantiene el espacio al final original
    Staying_with_Family: int
    Vacation_with_Family: int
    Family_Functions: int
    Disappointing_Thing: int
    Feelings: int
    Conflict: int
    Dissatisfaction: int
    Mental_Disturbances: int
    Politics: int
    Flexibility: int

@app.post("/predict")
def predict_burnout(data: BurnoutInput):
    # 1. Convertir los datos recibidos a un DataFrame de Pandas (usando los alias)
    input_data = data.dict(by_alias=True)
    df = pd.DataFrame([input_data])
    # Restaurar el espacio al final en la columna Gender tal como está en el modelo original
    df = df.rename(columns={"Gender": "Gender "})
    
    # 2. Obtener la probabilidad real de la clase 1 (Tiene Burnout)
    probability = float(model_pipeline.predict_proba(df)[0][1])
    
    # Definir el umbral de riesgo clínico
    UMBRAL_RIESGO = 0.60
    
    # 3. Determinar la predicción basada en nuestro nuevo umbral
    prediction = 1 if probability >= UMBRAL_RIESGO else 0
    
    top_3_factores = []
    
    # 4. Lógica para extraer los 3 factores principales si se detecta burnout (o riesgo)
    if prediction == 1:
        # Transformar los datos de entrada usando el OneHotEncoder del pipeline
        preprocessor = model_pipeline.named_steps['preprocessor']
        X_transformed = preprocessor.transform(df)
        feature_names_out = preprocessor.get_feature_names_out().tolist()
        
        # Extraer el modelo XGBoost y convertir los datos a DMatrix
        xgb_model = model_pipeline.named_steps['classifier']
        booster = xgb_model.get_booster()
        dmatrix = xgb.DMatrix(X_transformed, feature_names=feature_names_out)
        
        # Obtener las contribuciones de cada variable (Tree SHAP nativo de XGBoost)
        # El último valor es el "bias", lo omitimos con [:-1]
        contribuciones = booster.predict(dmatrix, pred_contribs=True)[0][:-1]
        
        importancia_original = {}
        
        # Agrupar las contribuciones de las variables dummy hacia la variable original
        for idx, col_name in enumerate(feature_names_out):
            # Limpiar el prefijo "cat__" y quitar el valor final "_X"
            nombre_real = col_name.replace("cat__", "").rsplit("_", 1)[0]
            contrib = contribuciones[idx]
            
            # Solo sumar las contribuciones POSITIVAS (las que aumentan el riesgo de Burnout)
            if contrib > 0:
                importancia_original[nombre_real] = importancia_original.get(nombre_real, 0) + contrib
                
        # Ordenar de mayor a menor contribución y tomar las 3 principales
        top_3 = sorted(importancia_original.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_factores = [item[0].strip() for item in top_3]

    # 5. Retornar la respuesta al frontend
    return {
        "prediction": prediction,
        "burnout_probability_percent": round(probability * 100, 2),
        "status": "Tiene Burnout" if prediction == 1 else "No Burnout",
        "top_3_influential_factors": top_3_factores
    }