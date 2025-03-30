"""
Enhanced pose and object detector for the CV game.
This module handles pose estimation, object detection, and facial expressions.
With support for different model combinations.
"""
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
import os
import tensorflow as tf

# Try to import YOLO, with a fallback
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: YOLOv8 not available. Using simplified object detection.")
    YOLO_AVAILABLE = False

class PoseObjectDetector:
    """
    Combined pose and object detector using MediaPipe or MoveNet for poses
    and YOLOv8 or SSD for object detection.
    """
    
    def __init__(self, pose_model="MediaPipe", object_model="YOLOv8"):
        """
        Initialize the detector with the specified models.
        
        Args:
            pose_model (str): "MediaPipe" or "MoveNet"
            object_model (str): "YOLOv8" or "SSD"
        """
        self.pose_model_type = pose_model
        self.object_model_type = object_model
        
        # Initialize pose detection model
        if pose_model == "MediaPipe":
            self._init_mediapipe_pose()
        elif pose_model == "MoveNet":
            self._init_movenet_pose()
        else:
            print(f"Unsupported pose model: {pose_model}, falling back to MediaPipe")
            self.pose_model_type = "MediaPipe"
            self._init_mediapipe_pose()
        
        # Initialize object detection model
        if object_model == "YOLOv8" and YOLO_AVAILABLE:
            self._init_yolo_object()
        elif object_model == "SSD":
            self._init_ssd_object()
        else:
            if object_model == "YOLOv8" and not YOLO_AVAILABLE:
                print("YOLOv8 not available, falling back to SSD")
            else:
                print(f"Unsupported object model: {object_model}, falling back to SSD")
            self.object_model_type = "SSD"
            self._init_ssd_object()
        
        # Initialize MediaPipe Face Detection
        self.mp_face = mp.solutions.face_detection
        self.face_model = self.mp_face.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1  # Use the full model (more accurate)
        )
        
        # Initialize MediaPipe Face Mesh for more detailed face landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_model = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Hands for palm detection
        self.mp_hands = mp.solutions.hands
        self.hands_model = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define object classes we're interested in
        self.target_objects = {
            "chair", "bottle", "book", "laptop", "cup", 
            "keyboard", "cell phone", "mouse", "remote", "couch",
            "potted plant", "tv", "bowl", "phone"  # Add "phone" as alias
        }
        
        # Current action and scene description
        self.current_action = "Standing"
        self.action_history = ["Standing"] * 5  # For smoothing
        
        # Facial expression states
        self.is_smiling = False
        self.eyes_closed = False
        self.showing_palm = False
        
        # For expression smoothing
        self.smile_history = [False] * 5
        self.eyes_closed_history = [False] * 5
        self.palm_history = [False] * 5
        
        # Calibration variables for facial expressions
        self.eye_height_values = []
        self.mouth_ratio_values = []
        self.is_calibrated = False
        self.calibration_frames = 0
        self.calibration_max_frames = 30
        self.eye_threshold = 0.015  # Initial value, will be calibrated
        self.smile_threshold = 4.0   # Initial value, will be calibrated
        
        # Benchmarking data
        self.pose_inference_time = 0
        self.object_inference_time = 0
        
        # Debug mode
        self.debug = True
        
        print(f"Detector initialized with pose={self.pose_model_type}, object={self.object_model_type}")
    
    def _init_mediapipe_pose(self):
        """Initialize MediaPipe pose detection."""
        self.mp_pose = mp.solutions.pose
        self.pose_model = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # Use medium complexity model
        )
    
    def _init_movenet_pose(self):
        """Initialize MoveNet pose detection."""
        try:
            # Try to load MoveNet model
            import tensorflow_hub as hub
            self.pose_model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            self.movenet_input_size = 256
        except Exception as e:
            print(f"Error loading MoveNet: {e}")
            print("Falling back to MediaPipe")
            self.pose_model_type = "MediaPipe"
            self._init_mediapipe_pose()
    
    def _init_yolo_object(self):
        """Initialize YOLOv8 object detection."""
        try:
            # Check if model file exists locally
            yolo_path = "yolov8n.pt"
            if not Path(yolo_path).exists():
                # This will trigger auto-download
                yolo_path = "yolov8n.pt"
            
            self.object_model = YOLO(yolo_path)
            print("YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Falling back to SSD")
            self.object_model_type = "SSD"
            self._init_ssd_object()
    
    def _init_ssd_object(self):
        """Initialize SSD MobileNet object detection."""
        try:
            # Try to load SSD MobileNet model
            model_path = "ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model"
            
            # If model doesn't exist, try to download it
            if not Path(model_path).exists():
                print(f"SSD model not found at {model_path}")
                try:
                    import tensorflow_hub as hub
                    # Try to load from TensorFlow Hub
                    temp_model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
                    # Save the model locally
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    tf.saved_model.save(temp_model, model_path)
                    print(f"Downloaded SSD model to {model_path}")
                except Exception as e:
                    print(f"Error downloading SSD model: {e}")
                    # Create a placeholder model that returns empty results
                    self.object_model = None
                    return
            
            # Load the model
            self.object_model = tf.saved_model.load(model_path).signatures['serving_default']
            print("SSD model loaded successfully")
        except Exception as e:
            print(f"Error loading SSD model: {e}")
            self.object_model = None
    
    def detect(self, frame):
        """
        Detect poses, faces, expressions, and objects in the frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            tuple: (pose_results, detected_objects)
        """
        # Convert to RGB for model input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Pose detection
        pose_start_time = time.time()
        if self.pose_model_type == "MediaPipe":
            pose_results = self._detect_mediapipe_pose(frame_rgb, frame)
        else:  # MoveNet
            pose_results = self._detect_movenet_pose(frame_rgb, frame)
        self.pose_inference_time = time.time() - pose_start_time
        
        # Face detection with MediaPipe
        face_results = self.face_model.process(frame_rgb)
        
        # Face mesh for detailed facial features
        face_mesh_results = self.face_mesh_model.process(frame_rgb)
        
        # Hand detection for palm gesture
        hands_results = self.hands_model.process(frame_rgb)
        
        # Update facial expressions
        self._detect_facial_expressions(frame_rgb, face_mesh_results, frame)
        
        # Draw hand landmarks if detected
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Detect palm showing gesture
                self._detect_palm_gesture(hand_landmarks, frame, w, h)
        
        # Object detection
        object_start_time = time.time()
        if self.object_model_type == "YOLOv8" and YOLO_AVAILABLE:
            detected_objects = self._detect_yolo_objects(frame_rgb, frame)
        else:  # SSD
            detected_objects = self._detect_ssd_objects(frame_rgb, frame)
        self.object_inference_time = time.time() - object_start_time
        
        # Add model info to frame
        self._add_model_info(frame)
        
        return pose_results, detected_objects
    
    def _detect_mediapipe_pose(self, frame_rgb, output_frame):
        """
        Detect pose using MediaPipe.
        
        Args:
            frame_rgb: RGB frame for processing
            output_frame: Frame to draw pose on
            
        Returns:
            MediaPipe pose results
        """
        results = self.pose_model.process(frame_rgb)
        
        # Draw pose landmarks if detected
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                output_frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        
        return results
    
    def _detect_movenet_pose(self, frame_rgb, output_frame):
        """
        Detect pose using MoveNet.
        
        Args:
            frame_rgb: RGB frame for processing
            output_frame: Frame to draw pose on
            
        Returns:
            MoveNet results converted to MediaPipe format for compatibility
        """
        h, w = output_frame.shape[:2]
        
        # Resize and convert image to tensor
        img = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 
                                    self.movenet_input_size, 
                                    self.movenet_input_size)
        img = tf.cast(img, dtype=tf.int32)
        
        # Run inference - fix for newer TensorFlow versions
        # Instead of: outputs = self.pose_model(img)
        # Use this:
        if hasattr(self.pose_model, 'signatures'):
            outputs = self.pose_model.signatures['serving_default'](tf.constant(img))
        else:
            outputs = self.pose_model(img)
            
        keypoints = outputs['output_0'].numpy()[0, 0, :, :3]  # Shape: [17, 3]
            # Convert to MediaPipe format for compatibility
        # Create a placeholder result
        class PlaceholderResult:
            def __init__(self):
                self.pose_landmarks = None
        
        result = PlaceholderResult()
        
        # Check if pose was detected (based on confidence of first keypoint)
        if keypoints[0, 2] > 0.3:
            # Create landmarks in MediaPipe format for compatibility
            landmarks = []
            for i in range(keypoints.shape[0]):
                landmark = type('', (), {})()  # Create empty object
                landmark.x = keypoints[i, 1]  # MoveNet returns [y, x, confidence]
                landmark.y = keypoints[i, 0]
                landmark.z = 0  # MoveNet doesn't provide z-coordinates
                landmark.visibility = keypoints[i, 2]  # Use confidence as visibility
                landmarks.append(landmark)
            
            # Create pose_landmarks
            result.pose_landmarks = type('', (), {})()
            result.pose_landmarks.landmark = landmarks
            
            # Draw keypoints on the frame
            for i, kp in enumerate(keypoints):
                if kp[2] > 0.3:  # Check confidence
                    x, y = int(kp[1] * w), int(kp[0] * h)
                    cv2.circle(output_frame, (x, y), 5, (0, 255, 0), -1)
            
            # Draw connections
            connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Face
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
            ]
            
            for connection in connections:
                if keypoints[connection[0], 2] > 0.3 and keypoints[connection[1], 2] > 0.3:
                    x0, y0 = int(keypoints[connection[0], 1] * w), int(keypoints[connection[0], 0] * h)
                    x1, y1 = int(keypoints[connection[1], 1] * w), int(keypoints[connection[1], 0] * h)
                    cv2.line(output_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
        return result
    
    def _detect_yolo_objects(self, frame_rgb, output_frame):
        """
        Detect objects using YOLOv8.
        
        Args:
            frame_rgb: RGB frame for processing
            output_frame: Frame to draw detections on
            
        Returns:
            List of detected objects
        """
        detected_objects = []
        
        # Run YOLOv8 detection
        try:
            results = self.object_model(frame_rgb, conf=0.3)  # Lower confidence threshold
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]
                    
                    # Handle cell phone / phone alias
                    if label.lower() == "cell phone":
                        label = "phone"  # Normalize to "phone" for easier task checking
                    
                    # Only keep objects we're interested in
                    if label.lower() in self.target_objects and conf > 0.3:
                        detected_objects.append({
                            "label": label.lower(),  # Normalize to lowercase
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "center": ((x1 + x2) // 2, (y1 + y2) // 2)
                        })
                        
                        # Draw bounding box
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label with background
                        text = f"{label} {conf:.2f}"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        
                        # Background rectangle for text
                        cv2.rectangle(
                            output_frame, 
                            (x1, y1 - text_size[1] - 5), 
                            (x1 + text_size[0], y1), 
                            (0, 255, 0), 
                            -1
                        )
                        
                        # Text
                        cv2.putText(
                            output_frame, 
                            text, 
                            (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 0, 0), 
                            2
                        )
        
        except Exception as e:
            print(f"Error in YOLO object detection: {e}")
        
        return detected_objects
    
    def _detect_ssd_objects(self, frame_rgb, output_frame):
        """
        Detect objects using SSD MobileNet.
        
        Args:
            frame_rgb: RGB frame for processing
            output_frame: Frame to draw detections on
            
        Returns:
            List of detected objects
        """
        detected_objects = []
        h, w = output_frame.shape[:2]
        
        if self.object_model is None:
            return detected_objects
        
        # Run SSD MobileNet detection
        try:
            # Prepare input tensor
            input_tensor = tf.convert_to_tensor(np.expand_dims(frame_rgb, 0), dtype=tf.uint8)
            
            # Run inference
            detections = self.object_model(input_tensor)
            
            # Process results
            num_detections = int(detections['num_detections'])
            
            # COCO label map
            coco_labels = {
                1: "person", 56: "chair", 57: "couch", 59: "potted plant", 
                62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 
                66: "keyboard", 67: "cell phone", 73: "book", 
                44: "bottle", 46: "wine glass", 74: "clock"
            }
            
            for i in range(num_detections):
                # Get detection confidence
                score = float(detections['detection_scores'][0, i].numpy())
                
                if score > 0.3:  # Confidence threshold
                    # Get class ID and convert to class name
                    class_id = int(detections['detection_classes'][0, i].numpy())
                    
                    if class_id in coco_labels:
                        label = coco_labels[class_id]
                        
                        # Handle cell phone / phone alias
                        if label.lower() == "cell phone":
                            label = "phone"
                        
                        # Only keep objects we're interested in
                        if label.lower() in self.target_objects:
                            # Get bounding box
                            bbox = detections['detection_boxes'][0, i].numpy()
                            y1, x1, y2, x2 = int(bbox[0] * h), int(bbox[1] * w), int(bbox[2] * h), int(bbox[3] * w)
                            
                            detected_objects.append({
                                "label": label.lower(),
                                "box": (x1, y1, x2, y2),
                                "conf": score,
                                "center": ((x1 + x2) // 2, (y1 + y2) // 2)
                            })
                            
                            # Draw bounding box
                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw label with background
                            text = f"{label} {score:.2f}"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            
                            # Background rectangle for text
                            cv2.rectangle(
                                output_frame, 
                                (x1, y1 - text_size[1] - 5), 
                                (x1 + text_size[0], y1), 
                                (0, 255, 0), 
                                -1
                            )
                            
                            # Text
                            cv2.putText(
                                output_frame, 
                                text, 
                                (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 0, 0), 
                                2
                            )
        
        except Exception as e:
            print(f"Error in SSD object detection: {e}")
        
        return detected_objects
    
    def _detect_facial_expressions(self, frame_rgb, face_mesh_results, display_frame):
        """
        Detect facial expressions (smile, eyes closed) from face mesh landmarks.
        With automatic calibration for better accuracy.
        
        Args:
            frame_rgb: RGB frame for processing
            face_mesh_results: Results from face mesh detection
            display_frame: Frame to draw visualizations on
        """
        h, w = display_frame.shape[:2]
        
        if not face_mesh_results.multi_face_landmarks:
            # Reset expression states when no face is detected
            self.is_smiling = False
            self.eyes_closed = False
            return
        
        # Get face landmarks
        face_landmarks = face_mesh_results.multi_face_landmarks[0]
        
        # Check for smile
        # Mouth landmark indices
        left_mouth = face_landmarks.landmark[61]
        right_mouth = face_landmarks.landmark[291]
        top_mouth = face_landmarks.landmark[0]
        bottom_mouth = face_landmarks.landmark[17]
        
        # Calculate mouth aspect ratio (width / height) - higher values indicate smile
        mouth_width = abs(right_mouth.x - left_mouth.x)
        mouth_height = abs(bottom_mouth.y - top_mouth.y)
        mouth_ratio = mouth_width / max(mouth_height, 0.01)  # Avoid division by zero
        
        # Check for eyes closed
        # Eye landmark indices
        left_eye_top = face_landmarks.landmark[159]
        left_eye_bottom = face_landmarks.landmark[145]
        right_eye_top = face_landmarks.landmark[386]
        right_eye_bottom = face_landmarks.landmark[374]
        
        # Calculate eye aspect ratios (height / width) - lower values indicate closed eyes
        left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
        right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        
        # Calibration logic
        if not self.is_calibrated and self.calibration_frames < self.calibration_max_frames:
            # Collect samples for calibration
            self.eye_height_values.append(avg_eye_height)
            self.mouth_ratio_values.append(mouth_ratio)
            self.calibration_frames += 1
            
            # When enough samples are collected, set the thresholds
            if self.calibration_frames >= self.calibration_max_frames:
                # Calculate average values
                avg_eye_height_normal = np.mean(self.eye_height_values)
                avg_mouth_ratio_normal = np.mean(self.mouth_ratio_values)
                
                # Set thresholds based on the normal state
                # Eye threshold: 70% of the average open eye height
                self.eye_threshold = avg_eye_height_normal * 0.7
                
                # Smile threshold: 120% of the normal mouth ratio
                self.smile_threshold = avg_mouth_ratio_normal * 1.2
                
                self.is_calibrated = True
                print(f"Calibration complete: Eye threshold: {self.eye_threshold:.4f}, Smile threshold: {self.smile_threshold:.2f}")
        
        # Use calibrated thresholds for detection
        current_smile = mouth_ratio > self.smile_threshold
        current_eyes_closed = avg_eye_height < self.eye_threshold
        
        # Draw debugging information about measurements
        cv2.putText(
            display_frame,
            f"Mouth ratio: {mouth_ratio:.2f}",
            (w - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        cv2.putText(
            display_frame,
            f"Eye height: {avg_eye_height:.4f}",
            (w - 200, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Smooth smile detection
        self.smile_history.pop(0)
        self.smile_history.append(current_smile)
        self.is_smiling = sum(self.smile_history) >= 3  # At least 3 out of 5 frames show smile
        
        # Smooth eyes closed detection
        self.eyes_closed_history.pop(0)
        self.eyes_closed_history.append(current_eyes_closed)
        self.eyes_closed = sum(self.eyes_closed_history) >= 3  # At least 3 out of 5 frames show closed eyes
        
        # Visualize expressions on the display frame
        if self.is_smiling:
            cv2.putText(
                display_frame,
                "Smiling",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        
        if self.eyes_closed:
            cv2.putText(
                display_frame,
                "Eyes Closed",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
    
    def _detect_palm_gesture(self, hand_landmarks, display_frame, width, height):
        """
        Detect if the person is showing their palm.
        
        Args:
            hand_landmarks: Hand landmarks from MediaPipe
            display_frame: Frame to draw visualizations on
            width: Frame width
            height: Frame height
        """
        # Check if fingers are extended
        # Get landmark coordinates normalized to image size
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height)))
        
        # Simple palm detection - check if fingers are extended
        # Finger indices:
        # Thumb: 1-4
        # Index: 5-8
        # Middle: 9-12
        # Ring: 13-16
        # Pinky: 17-20
        
        # Check if fingers are extended (tip is higher than knuckle)
        thumb_extended = landmarks[4][0] < landmarks[3][0]  # Thumb extends sideways
        index_extended = landmarks[8][1] < landmarks[5][1]
        middle_extended = landmarks[12][1] < landmarks[9][1]
        ring_extended = landmarks[16][1] < landmarks[13][1]
        pinky_extended = landmarks[20][1] < landmarks[17][1]
        
        # Palm is shown when most fingers are extended
        current_palm_showing = (index_extended and middle_extended and 
                                ring_extended and pinky_extended)
        
        # Smooth palm detection
        self.palm_history.pop(0)
        self.palm_history.append(current_palm_showing)
        self.showing_palm = sum(self.palm_history) >= 3  # At least 3 out of 5 frames show palm
        
        # Visualize palm detection
        if self.showing_palm:
            cv2.putText(
                display_frame,
                "Palm Showing",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
    
    def _add_model_info(self, frame):
        """
        Add model information overlay to the frame.
        
        Args:
            frame: Frame to add information to
        """
        h, w = frame.shape[:2]
        
        # Add model info at the bottom
        cv2.rectangle(frame, (0, h-60), (300, h), (0, 0, 0), -1)
        
        # Show model types
        cv2.putText(
            frame,
            f"Pose: {self.pose_model_type}",
            (10, h-40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        cv2.putText(
            frame,
            f"Object: {self.object_model_type}",
            (10, h-20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Show inference times
        cv2.putText(
            frame,
            f"Pose time: {self.pose_inference_time*1000:.1f}ms",
            (160, h-40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        cv2.putText(
            frame,
            f"Object time: {self.object_inference_time*1000:.1f}ms",
            (160, h-20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    def interpret_scene(self, pose_results, objects, frame_height):
        """
        Interpret the scene based on pose, facial expressions, and objects detected.
        
        Args:
            pose_results: Results from pose detection
            objects: List of detected objects
            frame_height: Height of the input frame
            
        Returns:
            str: Description of the scene
        """
        # Initialize with default description
        scene_description = "No person detected"
        
        # Handle different pose model formats
        if self.pose_model_type == "MediaPipe":
            pose_landmarks = pose_results.pose_landmarks if hasattr(pose_results, 'pose_landmarks') else None
        else:  # MoveNet
            pose_landmarks = pose_results.pose_landmarks if hasattr(pose_results, 'pose_landmarks') else None
        
        # Process pose if detected
        if pose_landmarks:
            landmarks = pose_landmarks.landmark
            
            # Calculate key points for pose classification
            if self.pose_model_type == "MediaPipe":
                shoulder_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
                
                hip_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y + 
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
                
                knee_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y + 
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y) / 2
            else:  # MoveNet (converted to MediaPipe format)
                # Get indices for MoveNet landmarks (may differ from MediaPipe)
                # Approximate mappings
                # MoveNet: 0=nose, 5-6=shoulders, 11-12=hips, 13-14=knees
                shoulder_y = (landmarks[5].y + landmarks[6].y) / 2
                hip_y = (landmarks[11].y + landmarks[12].y) / 2
                knee_y = (landmarks[13].y + landmarks[14].y) / 2
            
            # Determine action (standing, squatting, lifting)
            if knee_y > hip_y + 0.05:  # Knees higher than hips means squatting
                current_action = "Squatting"
            elif shoulder_y < hip_y - 0.1:  # Shoulders much lower than hips means lifting
                current_action = "Lifting"
            else:
                current_action = "Standing"
            
            # Smooth action detection using history
            self.action_history.pop(0)
            self.action_history.append(current_action)
            self.current_action = max(set(self.action_history), key=self.action_history.count)
            
            # Build scene description starting with pose
            scene_description = f"Person {self.current_action.lower()}"
            
            # Add facial expressions to scene description
            expression_parts = []
            if self.is_smiling:
                expression_parts.append("smiling")
            if self.eyes_closed:
                expression_parts.append("eyes closed")
            if self.showing_palm:
                expression_parts.append("showing palm")
                
            if expression_parts:
                scene_description += f", {' and '.join(expression_parts)}"
            
            # Find the nearest object
            if objects:
                # Convert hip_y from relative to absolute coordinates
                hip_y_abs = hip_y * frame_height
                
                # Sort objects by vertical distance to person's hip
                nearest_object = min(
                    objects, 
                    key=lambda obj: abs(obj["center"][1] - hip_y_abs)
                )
                
                # Add object to scene description
                if self.current_action == "Lifting" and nearest_object["label"] in ["chair", "bottle", "book", "phone"]:
                    scene_description = f"Person lifting a {nearest_object['label']}"
                    
                    # Add expressions if present
                    if expression_parts:
                        scene_description += f", {' and '.join(expression_parts)}"
                else:
                    scene_description += f" near a {nearest_object['label']}"
                
                if self.debug:
                    print(f"Scene description: {scene_description}")
            else:
                scene_description += ", no objects detected"
                
                if self.debug:
                    print(f"No objects detected. Scene description: {scene_description}")
        
        return scene_description