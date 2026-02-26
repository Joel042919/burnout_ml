from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Burnout Prediction API")

# Configurar CORS para permitir que tu frontend se comunique con la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, cambia "*" por la URL de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo guardado
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
    Liabilities: int = Field(alias="Liabilities ") # Nota: el string original tenía un espacio al final
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
    # Convertir los datos recibidos a un DataFrame de Pandas (usando los alias)
    input_data = data.dict(by_alias=True)
    df = pd.DataFrame([input_data])
    df = df.rename(columns={"Gender": "Gender "})
    
    # Hacer la predicción
    prediction = model_pipeline.predict(df)
    probability = model_pipeline.predict_proba(df)[0][1] # Probabilidad de tener burnout
    
    # 0 = No Burnout, 1 = Tiene Burnout
    return {
        "prediction": int(prediction[0]),
        "burnout_probability": float(probability),
        "status": "Tiene Burnout" if int(prediction[0]) == 1 else "No Burnout"
    }

# Para ejecutar el servidor, usa este comando en tu terminal:
# uvicorn main:app --reload