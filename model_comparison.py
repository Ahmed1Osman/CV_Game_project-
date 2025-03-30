"""
Model comparison application for evaluating different computer vision models.
"""
import os
import cv2
import time
import numpy as np
import threading
import argparse
from pathlib import Path

# Import project modules
from models.detector import PoseObjectDetector
from utils.benchmark import ModelBenchmark, create_model_matrix
from utils.visualization import draw_game_ui

class ModelComparisonApp:
    """
    Application for comparing performance of different computer vision models.
    """
    
    def __init__(self):
        """Initialize the model comparison application."""
        # Create detectors with different model combinations
        self.detectors = {
            "MediaPipe_YOLOv8": PoseObjectDetector(pose_model="MediaPipe", object_model="YOLOv8"),
            "MediaPipe_SSD": PoseObjectDetector(pose_model="MediaPipe", object_model="SSD"),
            "MoveNet_YOLOv8": PoseObjectDetector(pose_model="MoveNet", object_model="YOLOv8"),
            "MoveNet_SSD": PoseObjectDetector(pose_model="MoveNet", object_model="SSD")
        }
        
        # Create benchmark utility
        self.benchmark = ModelBenchmark()
        
        # Initialize frame storage for each model
        self.model_frames = {}
        self.model_results = {}
        
        # Control flags
        self.running = False
        self.compare_mode = False
        self.matrix_view = False
        self.current_model = "MediaPipe_YOLOv8"
        
        # Create output directory
        self.output_dir = Path("comparison_results")
        self.output_dir.mkdir(exist_ok=True)
    
    def start(self, camera_id=0):
        """
        Start the model comparison application.
        
        Args:
            camera_id (int): Camera device ID
        """
        self.running = True
        self.camera_id = camera_id
        
        # Start the camera thread
        self.camera_thread = threading.Thread(target=self._run_camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        # Start the display thread
        self.display_thread = threading.Thread(target=self._run_display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
        
        # Start benchmark session
        self.benchmark.start_session()
        
        # Wait for threads to complete
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.running = False
            if self.camera_thread.is_alive():
                self.camera_thread.join(timeout=1.0)
            if self.display_thread.is_alive():
                self.display_thread.join(timeout=1.0)
            
            # Save benchmark results
            report_file = self.benchmark.save_results()
            print(f"Saved benchmark report to: {report_file}")
    
    def _run_camera_loop(self):
        """Process frames from the camera."""
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            self.running = False
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"Camera opened successfully at {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        
        try:
            while self.running:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Update benchmark frame counter
                self.benchmark.update_frame_count()
                
                # Process frame with different models
                if self.compare_mode:
                    # Process with all models for comparison
                    for model_name, detector in self.detectors.items():
                        self._process_with_model(frame.copy(), model_name, detector)
                else:
                    # Process with only the current model
                    self._process_with_model(frame.copy(), self.current_model, self.detectors[self.current_model])
        
        finally:
            # Release camera
            cap.release()
    
    def _process_with_model(self, frame, model_name, detector):
        """
        Process a frame with a specific model.
        
        Args:
            frame: Input frame
            model_name: Name of the model combination
            detector: Detector instance
        """
        # Extract individual model names
        pose_model, object_model = model_name.split('_')
        
        # Run detection
        start_time = time.time()
        pose_results, objects = detector.detect(frame)
        inference_time = time.time() - start_time
        
        # Record metrics
        self.benchmark.record_inference_time(pose_model, inference_time / 2)  # Approximate time for pose
        self.benchmark.record_inference_time(object_model, inference_time / 2)  # Approximate time for object
        
        self.benchmark.record_detection_rate(pose_model, pose_results.pose_landmarks is not None)
        self.benchmark.record_detection_rate(object_model, len(objects) > 0)
        
        # Record object counts
        object_counts = {}
        for obj in objects:
            label = obj["label"]
            if label not in object_counts:
                object_counts[label] = 0
            object_counts[label] += 1
        
        self.benchmark.record_object_count(object_model, object_counts)
        
        # Interpret scene
        scene_description = detector.interpret_scene(pose_results, objects, frame.shape[0])
        
        # Draw UI elements
        draw_game_ui(frame, None, detector, scene_description)
        
        # Add benchmark visualization
        frame = self.benchmark.visualize_results(frame)
        
        # Store the processed frame
        self.model_frames[model_name] = frame
        self.model_results[model_name] = {
            "pose_results": pose_results,
            "objects": objects,
            "scene_description": scene_description,
            "inference_time": inference_time
        }
    
    def _run_display_loop(self):
        """Display the processed frames."""
        while self.running:
            # Skip if no frames are available yet
            if not self.model_frames:
                time.sleep(0.1)
                continue
            
            if self.compare_mode:
                if self.matrix_view and len(self.model_frames) > 1:
                    # Create matrix view of all models
                    frames = []
                    titles = []
                    for model_name in sorted(self.model_frames.keys()):
                        frames.append(self.model_frames[model_name])
                        titles.append(model_name)
                    
                    matrix = create_model_matrix(frames, titles)
                    if matrix is not None:
                        cv2.imshow("Model Comparison Matrix", matrix)
                else:
                    # Show side-by-side comparison of first two models
                    model_names = sorted(self.model_frames.keys())
                    if len(model_names) >= 2:
                        frame1 = self.model_frames[model_names[0]]
                        frame2 = self.model_frames[model_names[1]]
                        comparison = self.benchmark.compare_models(
                            frame1, frame2, model_names[0], model_names[1]
                        )
                        cv2.imshow("Model Comparison", comparison)
            else:
                # Show only the current model
                if self.current_model in self.model_frames:
                    cv2.imshow("Model Performance", self.model_frames[self.current_model])
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            self._handle_key_press(key)
            
            # Short delay to prevent excessive CPU usage
            time.sleep(0.01)
    
    def _handle_key_press(self, key):
        """
        Handle key presses.
        
        Args:
            key: Key code
        """
        if key == ord('q'):  # Quit
            self.running = False
        elif key == ord('c'):  # Toggle comparison mode
            self.compare_mode = not self.compare_mode
            print(f"Comparison mode: {self.compare_mode}")
        elif key == ord('m'):  # Toggle matrix view
            self.matrix_view = not self.matrix_view
            print(f"Matrix view: {self.matrix_view}")
        elif key == ord('s'):  # Save screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if self.compare_mode and self.matrix_view:
                frames = []
                titles = []
                for model_name in sorted(self.model_frames.keys()):
                    frames.append(self.model_frames[model_name])
                    titles.append(model_name)
                
                matrix = create_model_matrix(frames, titles)
                if matrix is not None:
                    filename = self.output_dir / f"matrix_comparison_{timestamp}.jpg"
                    cv2.imwrite(str(filename), matrix)
                    print(f"Saved matrix comparison to {filename}")
            elif self.compare_mode:
                model_names = sorted(self.model_frames.keys())
                if len(model_names) >= 2:
                    frame1 = self.model_frames[model_names[0]]
                    frame2 = self.model_frames[model_names[1]]
                    comparison = self.benchmark.compare_models(
                        frame1, frame2, model_names[0], model_names[1]
                    )
                    filename = self.output_dir / f"side_by_side_comparison_{timestamp}.jpg"
                    cv2.imwrite(str(filename), comparison)
                    print(f"Saved side-by-side comparison to {filename}")
            else:
                if self.current_model in self.model_frames:
                    filename = self.output_dir / f"{self.current_model}_{timestamp}.jpg"
                    cv2.imwrite(str(filename), self.model_frames[self.current_model])
                    print(f"Saved screenshot to {filename}")
        elif key == ord('r'):  # Save benchmark report
            report_file = self.benchmark.save_results()
            print(f"Saved benchmark report to: {report_file}")
        elif key == ord('1'):  # Switch to model combination 1
            self.current_model = "MediaPipe_YOLOv8"
            print(f"Switched to model: {self.current_model}")
        elif key == ord('2'):  # Switch to model combination 2
            self.current_model = "MediaPipe_SSD"
            print(f"Switched to model: {self.current_model}")
        elif key == ord('3'):  # Switch to model combination 3
            self.current_model = "MoveNet_YOLOv8"
            print(f"Switched to model: {self.current_model}")
        elif key == ord('4'):  # Switch to model combination 4
            self.current_model = "MoveNet_SSD"
            print(f"Switched to model: {self.current_model}")
        elif key == ord('b'):  # Cycle through models
            model_names = sorted(self.detectors.keys())
            current_index = model_names.index(self.current_model)
            next_index = (current_index + 1) % len(model_names)
            self.current_model = model_names[next_index]
            print(f"Switched to model: {self.current_model}")

def main():
    """Main function for the model comparison application."""
    parser = argparse.ArgumentParser(description="Model comparison for computer vision")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera index (default: 0)")
    parser.add_argument("--compare", action="store_true",
                       help="Start in comparison mode")
    parser.add_argument("--matrix", action="store_true",
                       help="Start in matrix view mode")
    args = parser.parse_args()
    
    # Create and start the comparison app
    app = ModelComparisonApp()
    app.compare_mode = args.compare
    app.matrix_view = args.matrix
    app.start(camera_id=args.camera)

if __name__ == "__main__":
    main()