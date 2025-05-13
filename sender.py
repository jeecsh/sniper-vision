import io
import socket
import struct
import time
import threading
import picamera
from queue import Queue

class PiCameraSender:
    def _init_(self, host='0.0.0.0', port=8000):
        self.host = host
        self.port = port
        self.camera = None
        self.server_socket = None
        self.connection = None
        self.frame_queue = Queue(maxsize=10)  # Buffer up to 10 frames
        self.running = False

    def start_camera(self):
        self.camera = picamera.PiCamera()
        self.camera.resolution = (640, 480)  # Balanced resolution for speed
        self.camera.framerate = 30
        time.sleep(2)  # Camera warm-up time

    def setup_server(self):
        self.server_socket = socket.socket()
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(0)
        print(f"Waiting for connection on port {self.port}...")
        self.connection = self.server_socket.accept()[0]
        print("Client connected!")

    def capture_frames(self):
        stream = io.BytesIO()
        
        for _ in self.camera.capture_continuous(stream, format='jpeg', use_video_port=True):
            if not self.running:
                break
                
            # Get frame data
            frame_data = stream.getvalue()
            
            # Clear stream for next frame
            stream.seek(0)
            stream.truncate()
            
            # Put frame in queue
            try:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame_data)
            except:
                pass

    def send_frames(self):
        try:
            while self.running:
                if not self.frame_queue.empty():
                    frame_data = self.frame_queue.get()
                    
                    # Send frame size followed by frame data
                    size = struct.pack('<L', len(frame_data))
                    self.connection.sendall(size + frame_data)
                    
        except (socket.error, ConnectionError) as e:
            print(f"Connection error: {e}")
            self.running = False

    def start(self):
        try:
            self.start_camera()
            self.setup_server()
            self.running = True
            
            # Start capture thread
            capture_thread = threading.Thread(target=self.capture_frames)
            capture_thread.start()
            
            # Start sending thread
            send_thread = threading.Thread(target=self.send_frames)
            send_thread.start()
            
            # Wait for threads
            capture_thread.join()
            send_thread.join()
            
        except KeyboardInterrupt:
            print("\nStopping server...")
        finally:
            self.cleanup()

    def cleanup(self):
        self.running = False
        if self.connection:
            self.connection.close()
        if self.server_socket:
            self.server_socket.close()
        if self.camera:
            self.camera.close()

if _name_ == '_main_':
    sender = PiCameraSender()
    sender.start()