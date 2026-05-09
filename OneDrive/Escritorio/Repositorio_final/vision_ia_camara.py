import cv2
import requests
import threading
import time
import os
from ultralytics import YOLO

class SeniorVisionSystem:
    def __init__(self, camera_index=0, lugar="Calle_Principal"):
        self.api_url = "http://localhost:8000/api/registros"
        self.lugar = lugar
        self.camera_index = camera_index
        
        print(f"[INFO] Inicializando IA en: {self.lugar}")
        self.yolo = YOLO('yolov8n.pt')
        
        self.gender_proto = "gender_deploy.prototxt"
        self.gender_model = "gender_net.caffemodel"
        self.gender_net = None
        
        if os.path.exists(self.gender_proto) and os.path.exists(self.gender_model):
            self.gender_net = cv2.dnn.readNet(self.gender_model, self.gender_proto)
            self.gender_list = ['masculino', 'femenino']
            print("[OK] Modelos de género cargados.")
        
        self.last_reg = 0
        self.cooldown = 4

        self.is_camera_running = False
        self.is_detection_running = False
        self.current_frame = None
        self.thread = None
        self.cap = None
        self.lock = threading.Lock()

    @staticmethod
    def list_available_cameras():
        available = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
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

    def process_and_send(self, clase, genero="N/A"):
        payload = {"clase": clase, "genero": genero, "lugar": self.lugar}
        try:
            requests.post(self.api_url, json=payload, timeout=2)
            print(f">>> EVENTO REGISTRADO: {clase.upper()} | GENERO: {genero} | UBICACIÓN: {self.lugar}")
        except:
            print("[!] Error: No se pudo conectar al Backend.")

    def start_camera(self):
        if not self.is_camera_running:
            self.is_camera_running = True
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._update_camera, daemon=True)
                self.thread.start()

    def stop_camera(self):
        print("[INFO] Deteniendo cámara. Esperando a que el hilo termine...")
        self.is_camera_running = False
        if self.thread is not None:
            if self.thread.is_alive():
                self.thread.join() # Espera total para evitar bloquear el hardware
            self.thread = None
        with self.lock:
            self.current_frame = None
        print("[INFO] Cámara detenida correctamente.")

    def toggle_detection(self):
        self.is_detection_running = not self.is_detection_running
        return self.is_detection_running

    def _update_camera(self):
        # Si es un entero, asumimos cámara USB local y usamos CAP_DSHOW en Windows.
        # Si es string (IP de Droidcam), omitimos CAP_DSHOW.
        if isinstance(self.camera_index, int):
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.camera_index)
        
        # Probar si realmente puede leer frames (a veces abre pero está bloqueada)
        can_read = False
        for _ in range(30): # Esperar hasta 3 segundos para que la cámara caliente
            try:
                ret, _ = cap.read()
                if ret:
                    can_read = True
                    break
            except Exception as e:
                print(f"[CRÍTICO] Error al leer cámara {self.camera_index}: {e}")
            time.sleep(0.1)

        if not cap.isOpened() or not can_read:
            print(f"[CRÍTICO] No se pudo obtener imagen de la cámara principal (0).")
            cap.release()
            self.is_camera_running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.is_camera_running:
            try:
                ret, frame = cap.read()
            except Exception as e:
                print(f"[ERROR OpenCV] Fallo al capturar frame: {e}")
                time.sleep(0.1)
                continue

            if not ret:
                time.sleep(0.1)
                continue

            if self.is_detection_running:
                results = self.yolo(frame, classes=[0, 15, 16], verbose=False)
                for r in results:
                    for box in r.boxes:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        cls = "persona" if int(box.cls[0]) == 0 else "animal"
                        
                        label = cls.capitalize()
                        color = (0, 255, 255) # Amarillo para animales
                        genero_detectado = "N/A"

                        if cls == "persona":
                            face = frame[by1:by2, bx1:bx2]
                            if face.size > 0:
                                genero_detectado = self.predict_gender(face)
                                if genero_detectado == "femenino":
                                    label = "Femenino"
                                    color = (180, 105, 255) # Rosa en BGR
                                elif genero_detectado == "masculino":
                                    label = "Masculino"
                                    color = (255, 144, 30) # Azul en BGR
                                else:
                                    label = "Persona"
                                    color = (255, 0, 255) # Magenta genérico
                        
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                        cv2.putText(frame, f"{label}", (bx1, by1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        if time.time() - self.last_reg > self.cooldown:
                            threading.Thread(target=self.process_and_send, 
                                             args=(cls, genero_detectado)).start()
                            self.last_reg = time.time()

            with self.lock:
                self.current_frame = frame.copy()

            time.sleep(0.03)

        # Liberar la cámara cuando se rompe el ciclo (is_camera_running == False)
        cap.release()

    def get_frame(self):
        with self.lock:
            if self.current_frame is None:
                return None
            ret, jpeg = cv2.imencode('.jpg', self.current_frame)
            return jpeg.tobytes() if ret else None