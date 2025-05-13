from flask import Flask, render_template
from flask_socketio import SocketIO
import socket
import struct
import cv2
import numpy as np
import base64

app = Flask(__name__)
socketio = SocketIO(app)

class StreamReceiver:
    def __init__(self, host='192.168.0.201', port=8000):
        self.host = host
        self.port = port
        self.client_socket = None
        self.running = False

    def connect(self):
        self.client_socket = socket.socket()
        print(f"Connecting to {self.host}:{self.port}...")
        self.client_socket.connect((self.host, self.port))
        print("Connected to server!")

    def receive_frame(self):
        # Read frame size
        size_data = self.client_socket.recv(struct.calcsize('<L'))
        if not size_data:
            return None
            
        frame_size = struct.unpack('<L', size_data)[0]
        
        # Read frame data
        frame_data = b''
        while len(frame_data) < frame_size:
            remaining = frame_size - len(frame_data)
            data = self.client_socket.recv(4096 if remaining > 4096 else remaining)
            if not data:
                return None
            frame_data += data
            
        return frame_data

    def process_frame(self, frame_data):
        # Convert frame data to numpy array
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        # Convert to base64 for sending to browser
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64

    def cleanup(self):
        self.running = False
        if self.client_socket:
            self.client_socket.close()

# Create global receiver instance
receiver = StreamReceiver()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    try:
        receiver.connect()
        socketio.start_background_task(target=stream_frames)
    except Exception as e:
        print(f"Connection error: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    receiver.cleanup()

def stream_frames():
    receiver.running = True
    while receiver.running:
        try:
            frame_data = receiver.receive_frame()
            if frame_data is None:
                break
                
            frame_base64 = receiver.process_frame(frame_data)
            socketio.emit('frame', {'image': frame_base64})
            
        except Exception as e:
            print(f"Streaming error: {e}")
            break
    
    receiver.cleanup()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
