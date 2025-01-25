from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import base64
from video_processor import process_frame

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


# Relay the video stream
@app.route("/")
def index():
    return "WebSocket API is live!"


@socketio.on("connect")
def handle_connect():
    print("Client connected!")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected!")


@socketio.on("raspberry-pi-frame")
def handle_raspberry_pi_frame(data):
    # Decode the received base64-encoded frame
    processed_frame = process_frame(data)
    # Broadcast the frame to all connected clients
    socketio.emit("video-frame", processed_frame)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
