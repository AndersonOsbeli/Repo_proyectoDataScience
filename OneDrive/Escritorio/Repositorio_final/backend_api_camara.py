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

from fastapi.responses import StreamingResponse
from vision_ia_camara import SeniorVisionSystem
import time
import cv2

app = FastAPI()
security = HTTPBearer()
# --- CONFIGURACIÓN ---
DB_FILE = "registro_transito.xlsx"
COLUMNS = ["id_registro", "clase", "genero", "fecha", "hora", "lugar"]
lock = filelock.FileLock(DB_FILE + ".lock")

app = FastAPI(title="Senior IA Traffic API")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite peticiones desde Angular (localhost:4200)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar Excel
if not os.path.exists(DB_FILE):
    pd.DataFrame(columns=COLUMNS).to_excel(DB_FILE, index=False)

# --- MODELOS ---
class RegistroSchema(BaseModel):
    clase: str # persona o animal
    genero: Optional[str] = None
    lugar: str
# --- SISTEMA DE VISION GLOBAL (una sola cámara activa) ---
vision_system = SeniorVisionSystem(camera_index=0, lugar="Cámara Principal")
vision_system.start_camera()

# --- VIDEO STREAM ---
def generate_video():
    import numpy as np
    blank = np.zeros((10, 10, 3), dtype=np.uint8)
    _, blank_jpeg = cv2.imencode('.jpg', blank)
    blank_bytes = blank_jpeg.tobytes()
    while True:
        frame = vision_system.get_frame()
        if frame is None:
            time.sleep(0.1)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/api/camera/stream")
async def video_feed():
    return StreamingResponse(generate_video(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/camera/status")
async def get_camera_status():
    return {
        "camera_running": vision_system.is_camera_running,
        "detection_running": vision_system.is_detection_running
    }

@app.post("/api/camera/toggle")
async def toggle_detection():
    vision_system.is_detection_running = not vision_system.is_detection_running
    return {
        "status": "ok",
        "camera_running": vision_system.is_camera_running,
        "detection_running": vision_system.is_detection_running
    }

class CameraSourceSchema(BaseModel):
    source: str  # "0" para PC, o URL completa para DroidCam

@app.post("/api/camera/source")
async def change_camera_source(data: CameraSourceSchema):
    try:
        new_source = int(data.source)
    except ValueError:
        new_source = data.source
    vision_system.stop_camera()
    vision_system.camera_index = new_source
    vision_system.start_camera()
    return {"status": "ok", "source": str(new_source)}

class LocationUpdateSchema(BaseModel):
    lugar: str

@app.get("/api/camera/location")
async def get_camera_location():
    return {"lugar": vision_system.lugar}

@app.post("/api/camera/location")
async def update_camera_location(data: LocationUpdateSchema):
    vision_system.lugar = data.lugar
    return {"status": "ok", "lugar": vision_system.lugar}


async def get_current_user(res: HTTPAuthorizationCredentials = Depends(security)):
    token = res.credentials
    try:
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

# @app.get("/api/registros")
# async def obtener_registros(user: dict = Depends(get_current_user)):
#     # Ahora esta ruta está PROTEGIDA
#     try:
#         df = pd.read_excel(DB_FILE)
#         return df.to_dict(orient="records")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/registros")
async def obtener_registros():
    # Ruta temporal SIN PROTECCIÓN para pruebas del dashboard
    try:
        df = pd.read_excel(DB_FILE)
        # Reemplazar valores NaN por cadenas vacías
        df = df.fillna("")
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/registros/recent")
async def obtener_registros_recientes(limit: int = 50):
    try:
        if not os.path.exists(DB_FILE):
            return []
        df = pd.read_excel(DB_FILE)
        df = df.fillna("")
        df_recent = df.tail(limit).iloc[::-1] # Obtener los últimos N y revertir (más reciente primero)
        return df_recent.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.delete("/api/registros/{id_registro}")
# async def eliminar_registro(id_registro: str, user: dict = Depends(get_current_user)):
#     # Solo un administrador logueado en Angular puede borrar
#     # ... tu lógica de eliminación aquí ...
#     return {"message": "Registro eliminado"}

@app.delete("/api/registros/{id_registro}")
async def eliminar_registro(id_registro: str):
    # Ruta temporal SIN PROTECCIÓN
    return {"message": "Registro eliminado"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)