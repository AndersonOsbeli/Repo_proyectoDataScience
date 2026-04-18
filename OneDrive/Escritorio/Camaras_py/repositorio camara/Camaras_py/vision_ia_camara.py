import cv2
import requests
import threading
import time
import os
import sys # Para manejar argumentos de consola
from ultralytics import YOLO

class SeniorVisionSystem:
    def __init__(self, camera_index=0, lugar="Calle_Principal"):
        self.api_url = "http://localhost:8000/api/registros"
        self.lugar = lugar
        
        print(f"[INFO] Inicializando IA en: {self.lugar}")
        self.yolo = YOLO('yolov8n.pt')
        
        self.gender_proto = "gender_deploy.prototxt"
        self.gender_model = "gender_net.caffemodel"
        self.gender_net = None
        
        # Cargar modelos de género
        if os.path.exists(self.gender_proto) and os.path.exists(self.gender_model):
            self.gender_net = cv2.dnn.readNet(self.gender_model, self.gender_proto)
            self.gender_list = ['masculino', 'femenino']
            print("[OK] Modelos de género cargados.")
        
        self.last_reg = 0
        self.cooldown = 4

    def list_available_cameras():
        """Escanea los primeros 5 índices para ver qué cámaras están conectadas."""
        available = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # CAP_DSHOW es más rápido en Windows
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def predict_gender(self, face_img):
        if self.gender_net is None: return "N/D"
        try:
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                        (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            self.gender_net.setInput(blob)
            preds = self.gender_net.forward()
            return self.gender_list[preds[0].argmax()]
        except:
            return "No identificado"

    def process_and_send(self, frame, x1, y1, x2, y2, clase):
        genero = "N/A"
        if clase == "persona":
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                genero = self.predict_gender(face)

        payload = {"clase": clase, "genero": genero, "lugar": self.lugar}
        try:
            requests.post(self.api_url, json=payload, timeout=2)
            print(f">>> EVENTO REGISTRADO: {clase.upper()} | GENERO: {genero} | UBICACIÓN: {self.lugar}")
        except:
            print("[!] Error: No se pudo conectar al Backend.")

    def run(self, source_index=0):
        # Intentar abrir la cámara seleccionada
        cap = cv2.VideoCapture(source_index, cv2.CAP_DSHOW)
        
        # Ajustar resolución para cámara externa (Full HD si lo soporta)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print(f"[CRÍTICO] No se pudo abrir la cámara {source_index}")
            return

        print(f">>> MONITORIZANDO CÁMARA {source_index}... Presiona 'q' para salir.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Error de señal. Reintentando...")
                time.sleep(2)
                continue

            results = self.yolo(frame, classes=[0, 15, 16], verbose=False)
            for r in results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    cls = "persona" if int(box.cls[0]) == 0 else "animal"
                    
                    # Dibujar UI neón
                    color = (255, 0, 255) if cls == "persona" else (0, 255, 255)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                    cv2.putText(frame, f"{cls}", (bx1, by1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if time.time() - self.last_reg > self.cooldown:
                        threading.Thread(target=self.process_and_send, 
                                         args=(frame.copy(), bx1, by1, bx2, by2, cls)).start()
                        self.last_reg = time.time()

            cv2.imshow(f"IA Vision - Fuente: {self.lugar}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- Lógica de Selección de Cámara ---
    print("--- BUSCANDO CÁMARAS CONECTADAS ---")
    cameras = SeniorVisionSystem.list_available_cameras()
    
    if not cameras:
        print("[ERROR] No hay cámaras detectadas.")
    else:
        print(f"Cámaras encontradas: {cameras}")
        print("0 = Integrada | 1 = Externa (usualmente)")
        
        # Puedes cambiar esto por el índice que desees usar
        sel = int(input("Selecciona el índice de la cámara a usar: "))
        nombre_lugar = input("Ingresa el nombre de la ubicación (ej: Calle_Frente): ")
        
        system = SeniorVisionSystem(camera_index=sel, lugar=nombre_lugar)
        system.run(source_index=sel)