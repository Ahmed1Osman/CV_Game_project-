"""
Script to launch the model comparison tool.
"""
from model_comparison import ModelComparisonApp

if __name__ == "__main__":
    print("Starting Computer Vision Model Comparison Tool")
    print("----------------------------------------------")
    print("Controls:")
    print("  - Press 'c' to toggle comparison mode")
    print("  - Press 'm' to toggle matrix view")
    print("  - Press 1-4 to select different model combinations:")
    print("    1: MediaPipe + YOLOv8")
    print("    2: MediaPipe + SSD")
    print("    3: MoveNet + YOLOv8")
    print("    4: MoveNet + SSD")
    print("  - Press 'b' to cycle through model combinations")
    print("  - Press 's' to save a screenshot")
    print("  - Press 'r' to save benchmark report")
    print("  - Press 'q' to quit")
    print("----------------------------------------------")
    
    # Create and start the comparison app
    app = ModelComparisonApp()
    app.start(camera_id=0)