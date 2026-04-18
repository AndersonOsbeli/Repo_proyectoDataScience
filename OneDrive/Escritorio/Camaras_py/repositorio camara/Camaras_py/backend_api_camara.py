from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import credentials, auth
import pandas as pd
import os
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List
from fastapi import FastAPI, HTTPException
import filelock

# Inicialización de Firebase Admin
# Reemplaza el nombre del archivo por el que descargaste
cred = credentials.Certificate("personasdashboard_firebase.json") 
firebase_admin.initialize_app(cred)

app = FastAPI()
security = HTTPBearer()
# --- CONFIGURACIÓN ---
DB_FILE = "registro_transito.xlsx"
COLUMNS = ["id_registro", "clase", "genero", "fecha", "hora", "lugar"]
lock = filelock.FileLock(DB_FILE + ".lock")

app = FastAPI(title="Senior IA Traffic API")

# Inicializar Excel
if not os.path.exists(DB_FILE):
    pd.DataFrame(columns=COLUMNS).to_excel(DB_FILE, index=False)

# --- MODELOS ---
class RegistroSchema(BaseModel):
    clase: str # persona o animal
    genero: Optional[str] = None
    lugar: str

async def get_current_user(res: HTTPAuthorizationCredentials = Depends(security)):
    token = res.credentials
    try:
        # Esto verifica el token contra los servidores de Google/Firebase
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=401, 
            detail=f"Token inválido o expirado: {str(e)}"
        )
# --- ENDPOINTS CRUD ---

@app.post("/api/registros")
async def crear_registro(data: RegistroSchema):
    """Registra una detección en el Excel."""
    try:
        with lock.acquire(timeout=10):
            df = pd.read_excel(DB_FILE)
            now = datetime.now()
            nuevo = {
                "id_registro": f"ID-{now.strftime('%M%S%f')[:-3]}",
                "clase": data.clase,
                "genero": data.genero or "N/A",
                "fecha": now.strftime("%Y-%m-%d"),
                "hora": now.strftime("%H:%M:%S"),
                "lugar": data.lugar
            }
            df = pd.concat([df, pd.DataFrame([nuevo])], ignore_index=True)
            df.to_excel(DB_FILE, index=False)
            return nuevo
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/registros")
async def obtener_registros(user: dict = Depends(get_current_user)):
    # Ahora esta ruta está PROTEGIDA
    try:
        df = pd.read_excel(DB_FILE)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/registros/{id_registro}")
async def eliminar_registro(id_registro: str, user: dict = Depends(get_current_user)):
    # Solo un administrador logueado en Angular puede borrar
    # ... tu lógica de eliminación aquí ...
    return {"message": "Registro eliminado"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)