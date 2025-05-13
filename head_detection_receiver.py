import socket
import struct
import cv2
import numpy as np
import time

class HighAccuracyForeheadTracker:
    def __init__(self, host='192.168.0.201', port=8000):
        self.host = host
        self.port = port
        self.client_socket = None
        self.running = False
        
        # Load YOLOv4-tiny head detection model
        self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        
        # Try to use GPU if available for faster processing
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using CUDA backend")
        except:
            try:
                if cv2.ocl.haveOpenCL():
                    cv2.ocl.setUseOpenCL(True)
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                    print("Using OpenCL backend")
                else:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    print("Using CPU backend")
            except:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Using CPU backend")

        # Load class names
        with open("head.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        
        # Enhanced optical flow parameters for fast movement
        self.prev_gray = None
        self.prev_pts = None
        self.track_points = None
        self.last_detection_time = 0
        self.detection_interval = 0.5  # Detect heads every 0.5 seconds
        
        # Improved optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),          # Larger window for better tracking
            maxLevel=4,                # More pyramid levels for faster movements
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            minEigThreshold=0.001      # Lower threshold for better point tracking
        )
        
        # Kalman filter for smooth tracking
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kalman_initialized = False
        
        # Create a smaller sniper scope overlay
        self.scope_size = 120  # Smaller scope size
        self.create_scope_overlay()
        
        # Display settings
        self.display_width = 1280
        self.display_height = 720

    def create_scope_overlay(self):
        # Create a smaller sniper scope overlay
        size = self.scope_size
        self.scope_overlay = np.zeros((size, size, 4), dtype=np.uint8)
        
        # Draw outer circle - thinner line
        cv2.circle(self.scope_overlay, (size//2, size//2), size//2-2, (0, 255, 0,200), 1)
        
        # Draw crosshair - thinner lines
        cv2.line(self.scope_overlay, (size//2, 5), (size//2, size//2-5), (0, 0, 0, 180), 1)
        cv2.line(self.scope_overlay, (size//2, size//2+5), (size//2, size-5), (0, 0, 0, 180), 1)
        cv2.line(self.scope_overlay, (5, size//2), (size//2-5, size//2), (0, 0, 0, 180), 1)
        cv2.line(self.scope_overlay, (size//2+5, size//2), (size-5, size//2), (0, 0, 0, 180), 1)
        
        # Draw center dot - smaller
        cv2.circle(self.scope_overlay, (size//2, size//2), 2, (0, 0, 255, 255), -1)
        
        # Add distance markers - fewer and thinner
        for i in range(1, 3):
            radius = (size//2) * i//3
            cv2.circle(self.scope_overlay, (size//2, size//2), radius, (0, 255, 0, 120), 1)

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

    def detect_heads(self, frame):
        # Resize frame to improve processing speed while maintaining aspect ratio
        target_height = 480
        ratio = target_height / frame.shape[0]
        target_width = int(frame.shape[1] * ratio)
        frame_resized = cv2.resize(frame, (target_width, target_height))
        
        # Create blob with smaller size for better performance
        blob = cv2.dnn.blobFromImage(frame_resized, 1/255.0, (320, 320), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []

        # Original frame dimensions for scaling boxes back
        height, width = frame.shape[:2]
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Higher confidence threshold for better precision
                if confidence > 0.65:  # Increased confidence threshold
                    # Scale coordinates back to original frame size
                    center_x = int((detection[0] * target_width) / ratio)
                    center_y = int((detection[1] * target_height) / ratio)
                    w = int((detection[2] * target_width) / ratio)
                    h = int((detection[3] * target_height) / ratio)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        # Non-max suppression with tighter thresholds
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.65, 0.3)
        final_boxes = []
        final_confidences = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                
        return final_boxes, final_confidences

    def forehead_position(self, box):
        """Calculate the precise forehead position from a head bounding box"""
        x, y, w, h = box
        
        # More precise forehead position based on anthropometric ratios
        # The forehead is approximately at the upper 15-20% of the head
        forehead_y = y + int(h * 0.15)  # 15% down from the top of the head box
        forehead_x = x + int(w * 0.5)   # Center of the head horizontally
        
        return (forehead_x, forehead_y)

    def init_kalman(self, x, y):
        """Initialize Kalman filter with initial position"""
        self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.kalman_initialized = True

    def update_tracking_points(self, frame, boxes=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        current_time = time.time()
        force_detection = False
        
        # If we haven't detected a head in a while, force detection
        if current_time - self.last_detection_time > self.detection_interval:
            force_detection = True
        
        # If this is the first frame, we need to reinitialize tracking points,
        # or if we're forcing a new detection
        if boxes is not None or self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 10 or force_detection:
            # Only detect heads if we don't already have boxes
            if boxes is None and force_detection:
                boxes, _ = self.detect_heads(frame)
                self.last_detection_time = current_time
            
            # Initialize tracking points focused on the forehead area
            if boxes:
                # Get the highest confidence box and calculate forehead position
                forehead_x, forehead_y = self.forehead_position(boxes[0])
                
                # Initialize Kalman filter
                if not self.kalman_initialized:
                    self.init_kalman(forehead_x, forehead_y)
                else:
                    # Update Kalman with new measurement
                    measured = np.array([[np.float32(forehead_x)], [np.float32(forehead_y)]])
                    self.kalman.correct(measured)
                
                # Create a denser grid of points around the forehead for tracking
                pts = []
                size = 25  # Larger size for better capture
                step = 3   # Smaller step for more points
                
                for i in range(forehead_x - size, forehead_x + size, step):
                    for j in range(forehead_y - size, forehead_y + size, step):
                        if 0 <= i < frame.shape[1] and 0 <= j < frame.shape[0]:
                            pts.append([i, j])
                
                self.prev_pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
                self.prev_gray = gray.copy()
                
                # Return the kalman-filtered position
                predicted = self.kalman.predict()
                return (predicted[0][0], predicted[1][0]), None
            else:
                return None, None
        
        # Calculate optical flow to track points
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )
        
        # Filter only the valid points
        valid_mask = status.ravel() == 1
        if np.sum(valid_mask) < 5:  # Need more points for reliable tracking
            self.prev_pts = None
            self.prev_gray = None
            return None, None
            
        valid_prev_pts = self.prev_pts[valid_mask]
        valid_next_pts = next_pts[valid_mask]
        
        # Calculate the average movement
        if len(valid_prev_pts) > 0:
            # Use RANSAC to filter outlier points for better accuracy in fast movements
            if len(valid_prev_pts) > 8:  # Need at least 8 points for RANSAC
                try:
                    _, mask = cv2.findHomography(valid_prev_pts.reshape(-1, 2), 
                                               valid_next_pts.reshape(-1, 2), 
                                               cv2.RANSAC, 
                                               5.0)  # 5 pixel threshold
                    
                    # Filter points based on RANSAC mask
                    if mask is not None:
                        mask = mask.ravel() == 1
                        if np.sum(mask) >= 5:  # At least 5 inliers
                            ransac_prev_pts = valid_prev_pts.reshape(-1, 2)[mask]
                            ransac_next_pts = valid_next_pts.reshape(-1, 2)[mask]
                            
                            # Calculate center from filtered points
                            center = np.mean(ransac_next_pts, axis=0)
                        else:
                            center = np.mean(valid_next_pts.reshape(-1, 2), axis=0)
                    else:
                        center = np.mean(valid_next_pts.reshape(-1, 2), axis=0)
                except:
                    center = np.mean(valid_next_pts.reshape(-1, 2), axis=0)
            else:
                center = np.mean(valid_next_pts.reshape(-1, 2), axis=0)
            
            # Update Kalman with new measurement
            if self.kalman_initialized:
                measured = np.array([[np.float32(center[0])], [np.float32(center[1])]])
                self.kalman.correct(measured)
                
                # Predict next position
                predicted = self.kalman.predict()
                smooth_center = (predicted[0][0], predicted[1][0])
            else:
                self.init_kalman(center[0], center[1])
                smooth_center = (center[0], center[1])
            
            # Update previous points and frame for next iteration
            self.prev_pts = valid_next_pts.reshape(-1, 1, 2)
            self.prev_gray = gray.copy()
            
            return smooth_center, None
        else:
            self.prev_pts = None
            self.prev_gray = None
            return None, None

    def overlay_sniper_scope(self, frame, center):
        if center is None:
            return frame
            
        x, y = int(center[0]), int(center[1])
        
        # Get scope dimensions
        h_scope, w_scope = self.scope_overlay.shape[:2]
        
        # Calculate position to place the scope centered on target
        top = max(0, y - h_scope // 2)
        left = max(0, x - w_scope // 2)
        
        # Make sure the scope stays within the frame boundaries
        h, w = frame.shape[:2]
        if top + h_scope > h:
            top = h - h_scope
        if left + w_scope > w:
            left = w - w_scope
            
        # Create a region of interest (ROI)
        roi = frame[top:top+h_scope, left:left+w_scope]
        
        # If the ROI has a different shape than the overlay, resize the overlay
        if roi.shape[0] != h_scope or roi.shape[1] != w_scope:
            # Get the actual available size
            actual_h = min(h_scope, h - top)
            actual_w = min(w_scope, w - left)
            
            # Create a new ROI with the correct size
            roi = frame[top:top+actual_h, left:left+actual_w]
            
            # Resize overlay to match the ROI
            resized_overlay = cv2.resize(self.scope_overlay, (actual_w, actual_h))
            
            # Blend the resized overlay with the ROI
            for c in range(0, 3):
                roi[:, :, c] = roi[:, :, c] * (1 - resized_overlay[:, :, 3]/255.0) + \
                               resized_overlay[:, :, c] * (resized_overlay[:, :, 3]/255.0)
        else:
            # Blend the scope overlay with the ROI
            for c in range(0, 3):
                roi[:, :, c] = roi[:, :, c] * (1 - self.scope_overlay[:, :, 3]/255.0) + \
                               self.scope_overlay[:, :, c] * (self.scope_overlay[:, :, 3]/255.0)
        
        # Draw thin targeting lines across the entire frame
        cv2.line(frame, (0, y), (w, y), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (x, 0), (x, h), (0, 255, 0), 1, cv2.LINE_AA)
        
        # Draw a small dot at the exact forehead position
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1, cv2.LINE_AA)
        
        return frame

    def start(self):
        try:
            self.connect()
            self.running = True
            
            # Create a large window for better visibility
            cv2.namedWindow('Precision Forehead Tracker', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Precision Forehead Tracker', self.display_width, self.display_height)
            
            while self.running:
                # Get frame data
                frame_data = self.receive_frame()
                if frame_data is None:
                    break
                    
                # Convert frame data to numpy array
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                # Fix any potential color issues
                frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR)
                
                # Start timing
                start_time = time.time()
                
                # Check if we need to perform detection
                current_time = time.time()
                if current_time - self.last_detection_time > self.detection_interval:
                    # Detect heads
                    boxes, confidences = self.detect_heads(frame)
                    self.last_detection_time = current_time
                    
                    # Update tracking with new detection
                    if boxes:
                        center, _ = self.update_tracking_points(frame, boxes)
                    else:
                        # If no detection, try to continue with existing tracking
                        center, _ = self.update_tracking_points(frame, None)
                else:
                    # Continue with optical flow tracking
                    center, _ = self.update_tracking_points(frame, None)
                
                # Overlay sniper scope on the tracked forehead
                if center is not None:
                    frame = self.overlay_sniper_scope(frame, center)
                    
                    # Display tracking info
                    fps = 1.0 / (time.time() - start_time)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display frame
                
                cv2.imshow('Precision Forehead Tracker', frame )
                
                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except (socket.error, ConnectionError) as e:
            print(f"Connection error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    tracker = HighAccuracyForeheadTracker()
    tracker.start()