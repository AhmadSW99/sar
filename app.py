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
from pymongo import MongoClient

# إعداد Flask
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# تحميل النموذج المخصص
model = YOLO('best.pt')  # <-- غير اسم الملف إذا كان مختلفًا

# إعداد الاتصال بـ MongoDB
MONGO_URI = os.environ.get("MONGO_URI")  # سنضيف هذا في Render
client = MongoClient(MONGO_URI)
db = client["ai-desert"]
collection = db["detections"]

# معالجة الصور عبر WebSocket
@socketio.on('upload_image')
def handle_upload_image(data):
    print("✅ Received image from client")

    try:
        image_data_base64 = data.get('image')
        if ',' in image_data_base64:
            _, image_data_base64 = image_data_base64.split(',', 1)

        img_bytes = base64.b64decode(image_data_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('error', {'message': 'Failed to decode image'})
            return

        result = model.predict(frame, verbose=False, conf=0.4)

        # رسم الصناديق
        annotated_frame = frame.copy()
        if result and result[0] and result[0].boxes:
            for box in result[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        # حفظ البيانات في قاعدة البيانات
        collection.insert_one({
            "timestamp": datetime.datetime.utcnow(),
            "boxes_detected": len(result[0].boxes) if result and result[0].boxes else 0,
            "image": processed_base64
        })

        emit('processed_image', {'image': f"data:image/jpeg;base64,{processed_base64}"})

    except Exception as e:
        print(f"❌ Error: {e}")
        emit('error', {'message': str(e)})

@socketio.on('connect')
def handle_connect():
    print('🔌 Client connected!')

@socketio.on('disconnect')
def handle_disconnect():
    print('⚠️ Client disconnected.')

@app.route('/')
def index():
    return "🟢 YOLOv8 WebSocket server with MongoDB is running."

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5005)
