# backend/app.py

import eventlet
eventlet.monkey_patch()

from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Load YOLO model
model = YOLO('yolov8n.pt')

@socketio.on('upload_image')
def handle_upload_image(data):
    print("Received image from client")

    try:
        image_data_base64 = data.get('image')

        # If there is a header like "data:image/jpeg;base64,..."
        if ',' in image_data_base64:
            _, image_data_base64 = image_data_base64.split(',', 1)

        img_bytes = base64.b64decode(image_data_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('error', {'message': 'Failed to decode image'})
            return

        # Process the frame
        result = model.predict(frame, verbose=False, conf=0.4)

        # Draw boxes
        annotated_frame = frame.copy()
        if result and result[0] and result[0].boxes:
            for box in result[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # Encode the result to base64 again
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        # Send the processed image back
        emit('processed_image', {'image': f"data:image/jpeg;base64,{processed_base64}"})

    except Exception as e:
        print(f"Error: {e}")
        emit('error', {'message': str(e)})

@socketio.on('connect')
def handle_connect():
    print('Client connected!')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected.')

@app.route('/')
def index():
    return "YOLOv8 Backend WebSocket server running."

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5005)
