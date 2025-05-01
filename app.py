import eventlet
eventlet.monkey_patch()

from flask import Flask, request
from flask_socketio import SocketIO, emit, disconnect # Import disconnect
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import datetime
import os
import requests
from pymongo import MongoClient
import traceback # Import traceback

# ===== Setup Flask App =====
app = Flask(__name__)
CORS(app)
# Increase timeouts slightly - main fix is async, but this adds buffer
# Default ping_timeout=5, ping_interval=25. Let's increase timeout.
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet", ping_timeout=20, ping_interval=25)

# ... (rest of your setup: download_yolo_from_drive, load model, MongoDB) ...

# Define the background processing task
def process_image_task(sid, image_data_base64):
    """Runs YOLO processing in a background thread."""
    print(f"🧵 [Task SID: {sid}] Starting background processing...")
    try:
        if model is None:
            # Emit error back to the specific client using their SID
            socketio.emit('error', {'message': 'YOLO model not loaded'}, room=sid)
            print(f"❌ [Task SID: {sid}] No model available.")
            return

        # Decode image
        if ',' in image_data_base64:
            _, img_b64_data = image_data_base64.split(',', 1)
        else:
            img_b64_data = image_data_base64 # Assume it's pure base64 if no comma

        img_bytes = base64.b64decode(img_b64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            socketio.emit('error', {'message': 'Failed to decode image'}, room=sid)
            print(f"❌ [Task SID: {sid}] Frame decoding failed.")
            return

        print(f"🎯 [Task SID: {sid}] Running YOLO prediction...")
        # Make sure model prediction itself doesn't hog resources indefinitely
        # Consider adding a timeout within the prediction call if possible,
        # or monitor resource usage closely.
        result = model.predict(frame, verbose=False, conf=0.4)
        print(f"✅ [Task SID: {sid}] Prediction completed.")

        # Draw boxes
        annotated_frame = frame.copy()
        num_boxes = 0
        if result and result[0] and hasattr(result[0], 'boxes') and result[0].boxes:
            num_boxes = len(result[0].boxes)
            print(f"✅ [Task SID: {sid}] Found {num_boxes} boxes.")
            for box in result[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
             print(f"ℹ️ [Task SID: {sid}] No boxes found or result structure unexpected.")

        print(f"🖼️ [Task SID: {sid}] Encoding processed image...")
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        print(f"✅ [Task SID: {sid}] Processed image encoded.")

        # Save to DB (Optional - keep it quick or make async too if needed)
        if collection:
            try:
                print(f"💾 [Task SID: {sid}] Attempting to save to MongoDB...")
                collection.insert_one({
                    "timestamp": datetime.datetime.utcnow(),
                    "boxes_detected": num_boxes,
                    "image_preview": processed_base64[:100] + "..." # Store preview
                })
                print(f"✅ [Task SID: {sid}] Image metadata saved to MongoDB.")
            except Exception as db_err:
                print(f"❌ [Task SID: {sid}] MongoDB Insert Failed: {db_err}")
        elif not collection:
             print(f"⚠️ [Task SID: {sid}] MongoDB collection not available. Skipping save.")

        # Emit result back ONLY to the original requester using their SID
        socketio.emit('processed_image', {'image': f"data:image/jpeg;base64,{processed_base64}"}, room=sid)
        print(f"📤 [Task SID: {sid}] Processed image sent to frontend.")

    except Exception as e:
        print(f"❌ [Task SID: {sid}] Error during background processing: {e}")
        traceback.print_exc() # Log full traceback
        # Emit error back to the specific client
        try:
            socketio.emit('error', {'message': f'Backend processing error: {str(e)}'}, room=sid)
        except Exception as emit_err:
             print(f"❌ [Task SID: {sid}] FAILED TO EMIT ERROR BACK TO CLIENT: {emit_err}")


# ===== WebSocket Upload Event (Modified) =====
@socketio.on('upload_image')
def handle_upload_image(data):
    # Get the Session ID of the client who sent the message
    sid = request.sid
    print(f"📥 [SID: {sid}] Received image from frontend.")

    image_data_base64 = data.get('image')
    if not image_data_base64:
         print(f"⚠️ [SID: {sid}] No image data received.")
         emit('error', {'message': 'No image data received.'}) # Emit directly as this is quick
         return

    # Start the heavy processing in a background task
    # Pass necessary data (like the image and the client's SID)
    socketio.start_background_task(process_image_task, sid, image_data_base64)

    # IMPORTANT: Return immediately! Do not wait for the task.
    # You could optionally emit a 'processing_started' message here if needed.
    # emit('processing_started', {'message': 'Image received, processing started.'})
    print(f"✅ [SID: {sid}] Handed off processing to background task. Handler returning.")
    # No 'return' statement needed, function just ends


# ... (rest of your Flask app: connect, disconnect, index, main) ...

# Make sure disconnect logs SID too
@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid # Get SID if available (might not be if disconnect abrupt)
    print(f"⚠️ Frontend disconnected. SID: {sid if sid else 'N/A'}")


if __name__ == "__main__":
    print("🚀 Starting Flask WebSocket server on port 5005...")
    # Use socketio.run for development with eventlet reload capabilities
    # For production with gunicorn/eventlet, the command is different
    socketio.run(app, host="0.0.0.0", port=5005, log_output=True) # Add log_output