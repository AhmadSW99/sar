import eventlet
eventlet.monkey_patch()

from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import datetime
import os
import requests
from pymongo import MongoClient

# ===== Setup Flask App =====
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ===== Function to download YOLOv8 from Google Drive =====
def download_yolo_from_drive(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    print(f"⬇️ Downloading YOLO model from Google Drive...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded successfully.")
    else:
        print(f"❌ Failed to download model. Status: {response.status_code}")

# ===== Load YOLO Model Safely =====
MODEL_PATH = "yolov8n.pt"
DRIVE_FILE_ID = "1T6MVvKaQ5VRQ6kXcypsn-UrFAyIivewW"  # <- Your model's file ID

try:
    if not os.path.exists(MODEL_PATH):
        download_yolo_from_drive(DRIVE_FILE_ID, MODEL_PATH)

    print("🟡 Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    print("✅ YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    model = None

# ===== Connect to MongoDB =====
try:
    MONGO_URI = os.environ.get("MONGO_URI")
    client = MongoClient(MONGO_URI)
    db = client["ai-desert"]
    collection = db["detections"]
    print("✅ Connected to MongoDB.")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    collection = None

# ===== WebSocket Upload Event =====
@socketio.on('upload_image')
def handle_upload_image(data):
    print("📥 Received image from frontend.")

    if model is None:
        emit('error', {'message': 'YOLO model not loaded'})
        print("❌ No model available.")
        return

    try:
        image_data_base64 = data.get('image')
        if ',' in image_data_base64:
            _, image_data_base64 = image_data_base64.split(',', 1)

        img_bytes = base64.b64decode(image_data_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('error', {'message': 'Failed to decode image'})
            print("❌ Frame decoding failed.")
            return

        print("🎯 Running YOLO prediction...")
        result = model.predict(frame, verbose=False, conf=0.4)
        print("✅ Prediction completed.")

        # Draw boxes
        annotated_frame = frame.copy()
        if result and result[0] and result[0].boxes:
            for box in result[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        # Save to DB
        if collection:
            collection.insert_one({
                "timestamp": datetime.datetime.utcnow(),
                "boxes_detected": len(result[0].boxes) if result and result[0].boxes else 0,
                "image": processed_base64
            })
            print("✅ Image and data saved to MongoDB.")

        emit('processed_image', {'image': f"data:image/jpeg;base64,{processed_base64}"})
        print("📤 Processed image sent to frontend.")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        emit('error', {'message': str(e)})

# ===== WebSocket Events =====
@socketio.on('connect')
def handle_connect():
    print("🔌 Frontend connected via WebSocket.")

@socketio.on('disconnect')
def handle_disconnect():
    print("⚠️ Frontend disconnected.")

# ===== Health Check Route =====
@app.route('/')
def index():
    return "🟢 YOLOv8 WebSocket server with MongoDB is running."

# ===== Start Server =====
if __name__ == "__main__":
    print("🚀 Starting Flask WebSocket server on port 5005...")
    socketio.run(app, host="0.0.0.0", port=5005)
