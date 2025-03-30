"""
Benchmark utility for comparing performance of different models.
"""
import time
import numpy as np
import csv
import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class ModelBenchmark:
    """
    Benchmark and compare performance of different computer vision models.
    """
    
    def __init__(self, output_dir="benchmark_results"):
        """
        Initialize the benchmark utility.
        
        Args:
            output_dir (str): Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance metrics for each model
        self.metrics = {
            # Pose detection models
            "MediaPipe": {
                "inference_times": [],
                "detection_rates": [],
                "task_success_rates": []
            },
            "MoveNet": {
                "inference_times": [],
                "detection_rates": [],
                "task_success_rates": []
            },
            # Object detection models
            "YOLOv8": {
                "inference_times": [],
                "detection_rates": [],
                "object_counts": {}
            },
            "SSD": {
                "inference_times": [],
                "detection_rates": [],
                "object_counts": {}
            }
        }
        
        # Model configurations to test
        self.configurations = [
            {"pose": "MediaPipe", "object": "YOLOv8"},
            {"pose": "MediaPipe", "object": "SSD"},
            {"pose": "MoveNet", "object": "YOLOv8"},
            {"pose": "MoveNet", "object": "SSD"}
        ]
        
        # Current configuration index
        self.current_config_index = 0
        
        # Benchmark session info
        self.session_start_time = None
        self.frames_processed = 0
        self.current_fps = 0
        self.comparison_mode = False
        
        # Results from current frame analysis
        self.current_results = {}
    
    def start_session(self, comparison_mode=False):
        """
        Start a new benchmark session.
        
        Args:
            comparison_mode (bool): Whether to run models side-by-side for comparison
        """
        self.session_start_time = time.time()
        self.frames_processed = 0
        self.comparison_mode = comparison_mode
        
        # Reset metrics for this session
        for model in self.metrics:
            self.metrics[model]["inference_times"] = []
            self.metrics[model]["detection_rates"] = []
            if "object_counts" in self.metrics[model]:
                self.metrics[model]["object_counts"] = {}
        
        print(f"Starting benchmark session at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Comparison mode: {comparison_mode}")
        
        if comparison_mode:
            print("Running all model configurations side by side")
        else:
            config = self.configurations[self.current_config_index]
            print(f"Testing configuration: Pose={config['pose']}, Object={config['object']}")
    
    def record_inference_time(self, model_name, inference_time):
        """
        Record inference time for a model.
        
        Args:
            model_name (str): Name of the model
            inference_time (float): Inference time in seconds
        """
        if model_name in self.metrics:
            self.metrics[model_name]["inference_times"].append(inference_time)
    
    def record_detection_rate(self, model_name, detected):
        """
        Record whether detection was successful.
        
        Args:
            model_name (str): Name of the model
            detected (bool): Whether detection was successful
        """
        if model_name in self.metrics:
            self.metrics[model_name]["detection_rates"].append(1 if detected else 0)
    
    def record_object_count(self, model_name, object_counts):
        """
        Record count of objects detected.
        
        Args:
            model_name (str): Name of the object detection model
            object_counts (dict): Dictionary of object counts by class
        """
        if model_name in self.metrics and "object_counts" in self.metrics[model_name]:
            for obj_class, count in object_counts.items():
                if obj_class not in self.metrics[model_name]["object_counts"]:
                    self.metrics[model_name]["object_counts"][obj_class] = []
                self.metrics[model_name]["object_counts"][obj_class].append(count)
    
    def record_task_success(self, model_name, success):
        """
        Record whether a task was successfully completed using the model.
        
        Args:
            model_name (str): Name of the model
            success (bool): Whether the task was completed successfully
        """
        if model_name in self.metrics and "task_success_rates" in self.metrics[model_name]:
            self.metrics[model_name]["task_success_rates"].append(1 if success else 0)
    
    def update_frame_count(self):
        """Update the frame count and calculate current FPS."""
        self.frames_processed += 1
        if self.session_start_time:
            elapsed_time = time.time() - self.session_start_time
            self.current_fps = self.frames_processed / elapsed_time
    
    def next_configuration(self):
        """Switch to the next model configuration."""
        if not self.comparison_mode:
            self.current_config_index = (self.current_config_index + 1) % len(self.configurations)
            config = self.configurations[self.current_config_index]
            print(f"Switched to configuration: Pose={config['pose']}, Object={config['object']}")
            return config
        return None
    
    def get_current_configuration(self):
        """Get the current model configuration."""
        return self.configurations[self.current_config_index]
    
    def save_results(self):
        """Save benchmark results to CSV files and generate reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw metrics
        metrics_file = self.output_dir / f"metrics_{timestamp}.csv"
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Metric", "Value"])
            
            for model, metrics in self.metrics.items():
                for metric, values in metrics.items():
                    if metric == "object_counts":
                        for obj_class, counts in values.items():
                            if counts:
                                writer.writerow([model, f"avg_{obj_class}_count", np.mean(counts)])
                    elif values:
                        writer.writerow([model, f"avg_{metric}", np.mean(values)])
                        writer.writerow([model, f"min_{metric}", np.min(values)])
                        writer.writerow([model, f"max_{metric}", np.max(values)])
        
        # Generate summary report
        report_file = self.output_dir / f"report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(f"Benchmark Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total frames processed: {self.frames_processed}\n")
            f.write(f"Average FPS: {self.current_fps:.2f}\n\n")
            
            f.write("Model Performance Summary:\n")
            f.write("==========================\n\n")
            
            for model, metrics in self.metrics.items():
                f.write(f"{model}:\n")
                if "inference_times" in metrics and metrics["inference_times"]:
                    avg_time = np.mean(metrics["inference_times"]) * 1000  # Convert to ms
                    f.write(f"  Average inference time: {avg_time:.2f} ms\n")
                
                if "detection_rates" in metrics and metrics["detection_rates"]:
                    avg_rate = np.mean(metrics["detection_rates"]) * 100  # Convert to percentage
                    f.write(f"  Detection rate: {avg_rate:.1f}%\n")
                
                if "task_success_rates" in metrics and metrics["task_success_rates"]:
                    avg_success = np.mean(metrics["task_success_rates"]) * 100  # Convert to percentage
                    f.write(f"  Task success rate: {avg_success:.1f}%\n")
                
                if "object_counts" in metrics:
                    f.write("  Average objects detected:\n")
                    for obj_class, counts in metrics["object_counts"].items():
                        if counts:
                            avg_count = np.mean(counts)
                            f.write(f"    {obj_class}: {avg_count:.1f}\n")
                
                f.write("\n")
        
        # Generate comparison charts
        self._generate_comparison_charts(timestamp)
        
        print(f"Benchmark results saved to {self.output_dir}")
        return report_file
    
    def _generate_comparison_charts(self, timestamp):
        """
        Generate comparison charts for the models.
        
        Args:
            timestamp (str): Timestamp for the filename
        """
        plt.figure(figsize=(12, 8))
        
        # Plot inference times
        plt.subplot(2, 2, 1)
        model_names = []
        avg_times = []
        
        for model, metrics in self.metrics.items():
            if "inference_times" in metrics and metrics["inference_times"]:
                model_names.append(model)
                avg_times.append(np.mean(metrics["inference_times"]) * 1000)  # Convert to ms
        
        plt.bar(model_names, avg_times)
        plt.title('Average Inference Time (ms)')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45)
        
        # Plot detection rates
        plt.subplot(2, 2, 2)
        model_names = []
        detection_rates = []
        
        for model, metrics in self.metrics.items():
            if "detection_rates" in metrics and metrics["detection_rates"]:
                model_names.append(model)
                detection_rates.append(np.mean(metrics["detection_rates"]) * 100)
        
        plt.bar(model_names, detection_rates)
        plt.title('Detection Rate (%)')
        plt.ylabel('Rate (%)')
        plt.xticks(rotation=45)
        
        # Plot task success rates for pose models
        plt.subplot(2, 2, 3)
        pose_models = []
        success_rates = []
        
        for model, metrics in self.metrics.items():
            if model in ["MediaPipe", "MoveNet"] and "task_success_rates" in metrics and metrics["task_success_rates"]:
                pose_models.append(model)
                success_rates.append(np.mean(metrics["task_success_rates"]) * 100)
        
        if pose_models:
            plt.bar(pose_models, success_rates)
            plt.title('Task Success Rate (%)')
            plt.ylabel('Rate (%)')
            plt.xticks(rotation=45)
        
        # Plot object detection counts
        plt.subplot(2, 2, 4)
        
        object_models = ["YOLOv8", "SSD"]
        object_classes = set()
        
        # Find all unique object classes
        for model in object_models:
            if model in self.metrics and "object_counts" in self.metrics[model]:
                object_classes.update(self.metrics[model]["object_counts"].keys())
        
        if object_classes:
            object_classes = list(object_classes)
            model_data = []
            
            for model in object_models:
                if model in self.metrics and "object_counts" in self.metrics[model]:
                    counts = []
                    for obj_class in object_classes:
                        if obj_class in self.metrics[model]["object_counts"] and self.metrics[model]["object_counts"][obj_class]:
                            counts.append(np.mean(self.metrics[model]["object_counts"][obj_class]))
                        else:
                            counts.append(0)
                    model_data.append(counts)
            
            x = np.arange(len(object_classes))
            width = 0.35
            
            if len(model_data) > 0:
                plt.bar(x - width/2, model_data[0], width, label=object_models[0])
            if len(model_data) > 1:
                plt.bar(x + width/2, model_data[1], width, label=object_models[1])
            
            plt.xlabel('Object Class')
            plt.ylabel('Average Count')
            plt.title('Object Detection Comparison')
            plt.xticks(x, object_classes, rotation=45)
            plt.legend()
        
        plt.tight_layout()
        chart_file = self.output_dir / f"comparison_chart_{timestamp}.png"
        plt.savefig(chart_file)
        plt.close()
    
    def visualize_results(self, frame, results=None):
        """
        Add visualization of benchmark results to the frame.
        
        Args:
            frame: The frame to visualize on
            results: Optional detection results to visualize
        
        Returns:
            The frame with visualizations
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-300, 0), (w, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add benchmark metrics
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        # Current configuration
        config = self.configurations[self.current_config_index]
        cv2.putText(frame, f"Pose: {config['pose']}", (w-290, y_offset), font, 0.6, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame, f"Object: {config['object']}", (w-290, y_offset), font, 0.6, (255, 255, 255), 1)
        y_offset += 30
        
        # FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w-290, y_offset), font, 0.6, (0, 255, 0), 2)
        y_offset += 25
        
        # Inference times
        if config['pose'] in self.metrics and self.metrics[config['pose']]["inference_times"]:
            avg_time = np.mean(self.metrics[config['pose']]["inference_times"]) * 1000  # ms
            cv2.putText(frame, f"Pose time: {avg_time:.1f} ms", (w-290, y_offset), font, 0.6, (255, 255, 255), 1)
            y_offset += 25
        
        if config['object'] in self.metrics and self.metrics[config['object']]["inference_times"]:
            avg_time = np.mean(self.metrics[config['object']]["inference_times"]) * 1000  # ms
            cv2.putText(frame, f"Object time: {avg_time:.1f} ms", (w-290, y_offset), font, 0.6, (255, 255, 255), 1)
            y_offset += 25
        
        # Frames processed
        cv2.putText(frame, f"Frames: {self.frames_processed}", (w-290, y_offset), font, 0.6, (255, 255, 255), 1)
        
        # Add instructions at the bottom
        cv2.putText(frame, "Press 'B' to switch model configuration", (10, h-40), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'R' to save benchmark report", (10, h-20), font, 0.6, (255, 255, 255), 1)
        
        return frame

    def compare_models(self, frame1, frame2, title1, title2):
        """
        Create a side-by-side comparison of two model outputs.
        
        Args:
            frame1: First frame
            frame2: Second frame
            title1: Title for first frame
            title2: Title for second frame
            
        Returns:
            Combined frame with both outputs
        """
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Resize frames to the same height if needed
        if h1 != h2:
            scale = h1 / h2
            new_w2 = int(w2 * scale)
            frame2 = cv2.resize(frame2, (new_w2, h1))
            w2 = new_w2
        
        # Create combined frame
        combined = np.zeros((h1, w1 + w2, 3), dtype=np.uint8)
        combined[:, :w1] = frame1
        combined[:, w1:w1+w2] = frame2
        
        # Add titles
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, title1, (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, title2, (w1 + 10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Add separator line
        cv2.line(combined, (w1, 0), (w1, h1), (255, 255, 255), 2)
        
        return combined

def create_model_matrix(frames, titles):
    """
    Create a matrix of frames for comparing multiple models.
    
    Args:
        frames: List of frames to arrange in a matrix
        titles: List of titles for each frame
        
    Returns:
        Matrix of frames
    """
    # Determine grid size
    count = len(frames)
    if count <= 2:
        cols = count
        rows = 1
    else:
        cols = 2
        rows = (count + 1) // 2
    
    # Make sure all frames have the same size
    if frames:
        target_h, target_w = frames[0].shape[:2]
        for i in range(1, len(frames)):
            frames[i] = cv2.resize(frames[i], (target_w, target_h))
        
        # Create blank canvas
        matrix_h = target_h * rows
        matrix_w = target_w * cols
        matrix = np.zeros((matrix_h, matrix_w, 3), dtype=np.uint8)
        
        # Place frames in the matrix
        for i, frame in enumerate(frames):
            if i < len(frames):
                row = i // cols
                col = i % cols
                y_start = row * target_h
                x_start = col * target_w
                matrix[y_start:y_start+target_h, x_start:x_start+target_w] = frame
                
                # Add title
                if i < len(titles):
                    cv2.putText(matrix, titles[i], (x_start + 10, y_start + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return matrix
    
    return None