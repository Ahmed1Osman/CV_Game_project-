# Computer Vision Interactive Game with Model Comparison

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
![OpenCV](https://img.shields.io/badge/opencv-4.7.0-green)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.10+-orange)

An interactive game that uses computer vision to detect poses, facial expressions, and objects in real-time while allowing users to benchmark and compare different CV models.

![Game Demo](https://via.placeholder.com/800x400?text=Game+Demo+Screenshot)

## Detection Examples

| Model | Detection Example |
|-------|------------------|
| MediaPipe + YOLOv8 | ![MediaPipe + YOLOv8](https://via.placeholder.com/400x300?text=MediaPipe+YOLOv8+Detection) |
| MediaPipe + SSD | ![MediaPipe + SSD](https://via.placeholder.com/400x300?text=MediaPipe+SSD+Detection) |
| MoveNet + YOLOv8 | ![MoveNet + YOLOv8](https://via.placeholder.com/400x300?text=MoveNet+YOLOv8+Detection) |
| MoveNet + SSD | ![MoveNet + SSD](https://via.placeholder.com/400x300?text=MoveNet+SSD+Detection) |

## Features

- **Pose Detection**: Recognize body poses using MediaPipe or MoveNet
- **Object Detection**: Identify objects with YOLOv8 or SSD MobileNet
- **Facial Expression Recognition**: Detect smiles and eye closure
- **Hand Gesture Recognition**: Detect open palm and other hand positions
- **Model Comparison Tool**: Benchmark and visualize performance differences between models
- **Real-time Performance Metrics**: View FPS, inference time, and detection accuracy
- **Screenshot Capability**: Save and organize gameplay images
- **Customizable Difficulty Levels**: Easy, medium, and hard game modes

## Model Comparison

This project uniquely allows you to compare the performance of different computer vision models:

| Model Type | Options | Strength |
|------------|---------|----------|
| Pose Detection | MediaPipe, MoveNet | Compare accuracy vs speed |
| Object Detection | YOLOv8, SSD MobileNet | Compare detection quality vs resource usage |

![Model Comparison](https://via.placeholder.com/800x400?text=Model+Comparison+Screenshot)

### Detection Characteristics

| Model | Strength | Weakness | Best Use Case |
|-------|----------|----------|---------------|
| MediaPipe Pose | Fast, lower resource usage | Less accurate for complex poses | Real-time applications, weaker hardware |
| MoveNet Pose | More accurate detection | Higher resource requirements | Detailed pose analysis, stronger hardware |
| YOLOv8 Object | Excellent accuracy, detects small objects | Higher GPU requirements | When detection quality is critical |
| SSD MobileNet | Resource efficient | Less accurate for distant/small objects | Mobile devices, CPU-only systems |

## Installation

### Prerequisites

- Python 3.8 or newer
- Webcam or camera device
- GPU recommended but not required

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cv-game-project.git
   cd cv-game-project
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download model files (automatically done on first run or can be done manually):
   ```bash
   python download_models.py  # Optional - models will download on first use if not present
   ```

## Usage

### Main Game

Run the main game interface:

```bash
python main.py
```

Controls:
- **Start Game**: Begin playing with selected difficulty
- **Stop Game**: End the current game
- **Take Screenshot**: Save the current frame
- **Switch Model**: Cycle through different model combinations

### Model Comparison Tool

To compare different computer vision models:

```bash
python compare_models.py
```

Keyboard controls:
- **C**: Toggle comparison mode (side-by-side view)
- **M**: Toggle matrix view (all models at once)
- **1-4**: Switch between model combinations
- **B**: Cycle through model combinations
- **S**: Save screenshot
- **R**: Generate and save benchmark report
- **Q**: Quit

## Game Tasks

The game challenges you to complete tasks that are detected through computer vision:

- **Easy mode**: Simple tasks like "stand", "squat", "smile", "near bottle"
- **Medium mode**: Combined tasks like "stand near bottle", "smile near chair"
- **Hard mode**: Complex tasks like "lift bottle", "squat then stand"

## Performance Considerations

- **CPU Usage**: MediaPipe + SSD combination is most efficient for CPU-only systems
- **GPU Performance**: YOLOv8 benefits significantly from GPU acceleration
- **Memory Usage**: MoveNet requires more memory than MediaPipe
- **Frame Rate**: Expect 15-30 FPS on mid-range hardware, depending on selected models

## Project Structure

```
cv_game_project/
├── models/           # Detection models
│   ├── detector.py   # Core detection logic
├── game/             # Game logic
│   ├── engine.py     # Game mechanics
├── utils/            # Utility functions
│   ├── benchmark.py  # Performance testing
│   ├── visualization.py # UI drawing
│   ├── screenshot.py # Screenshot functionality
├── main.py           # Main application entry point
├── compare_models.py # Model comparison tool
└── requirements.txt  # Dependencies
```

## Technical Details

### Pose Detection

- **MediaPipe**: Google's efficient pose estimation library
- **MoveNet**: TensorFlow's lightning-fast pose detection model

### Object Detection

- **YOLOv8**: Latest version of the YOLO (You Only Look Once) real-time object detector
- **SSD MobileNet**: Lightweight and efficient single-shot detection model

### Technologies Used

- **OpenCV**: For image processing and visualization
- **TensorFlow**: For deep learning models (MoveNet, SSD)
- **MediaPipe**: For efficient pose and face landmark detection
- **Ultralytics YOLOv8**: For state-of-the-art object detection
- **Tkinter**: For the user interface

## Extending the Project

Want to contribute? Here are some areas for improvement:

- Add more pose or object detection models for comparison
- Implement recording functionality to save video clips
- Create additional game modes or tasks
- Optimize for mobile or embedded devices
- Add multiplayer support

## Troubleshooting

### Common Issues

- **Camera not found**: Check your camera connection and ID (default is 0)
- **Model loading errors**: Ensure you have internet access for first-time model downloads
- **Low performance**: Try switching to lighter models (MediaPipe + SSD)
- **Detection issues**: Ensure good lighting and clear background

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for their excellent pose estimation library
- Ultralytics for YOLOv8 implementation
- TensorFlow team for MoveNet and SSD implementations
- OpenCV community for their comprehensive computer vision tools

---

## Citation

If you use this project in your research or work, please cite:

```
@software{cv_game_project,
  author = {Your Name},
  title = {Computer Vision Interactive Game with Model Comparison},
  year = {2025},
  url = {https://github.com/yourusername/cv-game-project}
}
```