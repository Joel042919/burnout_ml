from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import xgboost as xgb  # Importante importar xgboost

app = FastAPI(title="Burnout Prediction API")

# Configurar CORS para permitir que tu frontend (ej. React/Next.js) se comunique
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo guardado
model_pipeline = joblib.load("burnout_model.pkl")

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
    Liabilities: int = Field(alias="Liabilities ") 
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
    # 1. Preparar el DataFrame
    input_data = data.dict(by_alias=True)
    df = pd.DataFrame([input_data])
    df = df.rename(columns={"Gender": "Gender "})
    
    # 2. Hacer la predicción y obtener probabilidad real
    prediction = int(model_pipeline.predict(df)[0])
    probability = float(model_pipeline.predict_proba(df)[0][1]) 
    
    top_3_factores = []
    
    # 3. Lógica para extraer los 3 factores si tiene burnout
    if prediction == 1:
        # Transformar los datos de entrada usando el OneHotEncoder del pipeline
        preprocessor = model_pipeline.named_steps['preprocessor']
        X_transformed = preprocessor.transform(df)
        feature_names_out = preprocessor.get_feature_names_out()
        
        # Extraer el modelo XGBoost y convertir los datos a DMatrix
        xgb_model = model_pipeline.named_steps['classifier']
        booster = xgb_model.get_booster()
        dmatrix = xgb.DMatrix(X_transformed, feature_names=feature_names_out)
        
        # Obtener las contribuciones de cada variable (Tree SHAP nativo de XGBoost)
        # Esto devuelve un array donde el último valor es el "bias" general, lo omitimos ([:-1])
        contribuciones = booster.predict(dmatrix, pred_contribs=True)[0][:-1]
        
        importancia_original = {}
        
        # Como el OneHotEncoder divide una columna en varias (ej. cat__Working_Hour_6),
        # agrupamos la contribución sumando todo al nombre original de la variable.
        for idx, col_name in enumerate(feature_names_out):
            # Limpiar el prefijo "cat__" y quitar el valor final "_X" para obtener el nombre original
            nombre_real = col_name.replace("cat__", "").rsplit("_", 1)[0]
            contrib = contribuciones[idx]
            
            # Solo nos interesan las contribuciones POSITIVAS (las que empujaron al paciente hacia el Burnout)
            if contrib > 0:
                importancia_original[nombre_real] = importancia_original.get(nombre_real, 0) + contrib
                
        # Ordenar las variables por su contribución de mayor a menor y tomar las primeras 3
        top_3 = sorted(importancia_original.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_factores = [item[0].strip() for item in top_3]

    # 4. Retornar la respuesta estructurada
    return {
        "prediction": prediction,
        "burnout_probability_percent": round(probability * 100, 2),
        "status": "Tiene Burnout" if prediction == 1 else "No Burnout",
        "top_3_influential_factors": top_3_factores
    }