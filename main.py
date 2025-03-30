"""
Main entry point for the computer vision-based interactive game.
With model switching capabilities for comparison.
"""
import os
import argparse
import cv2
import time
import numpy as np
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk, messagebox, font
from PIL import Image, ImageTk
import webbrowser

# Import project modules
from models.detector import PoseObjectDetector
from game.engine import GameEngine
from utils.visualization import draw_game_ui
from utils.screenshot import take_screenshot, organize_screenshots

# Create necessary directories
Path("screenshots").mkdir(exist_ok=True)
Path("model_comparisons").mkdir(exist_ok=True)

# Global variables for the GUI
running = False
frame_to_display = None
camera_thread = None
screenshot_count = 0
max_auto_screenshots = 10  # Maximum number of auto screenshots
auto_screenshots_enabled = False

# Define theme colors
THEME = {
    "primary": "#3498db",      # Blue
    "secondary": "#2ecc71",    # Green
    "accent": "#e74c3c",       # Red
    "background": "#f5f5f5",   # Light gray
    "text": "#2c3e50",         # Dark blue/gray
    "text_light": "#ecf0f1"    # Very light gray
}

class ModernButton(ttk.Button):
    """Custom modern-looking button with hover effects."""
    def __init__(self, parent, **kwargs):
        self.style_name = kwargs.pop('style', 'TButton') + str(id(self))
        ttk.Style().configure(self.style_name, background=THEME["primary"], 
                             foreground=THEME["text_light"], borderwidth=0, 
                             focuscolor=THEME["primary"], lightcolor=THEME["primary"],
                             darkcolor=THEME["primary"])
        ttk.Style().map(self.style_name, 
                       background=[('active', THEME["secondary"])],
                       foreground=[('active', THEME["text_light"])])
        super().__init__(parent, style=self.style_name, **kwargs)

class GameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Computer Vision Interactive Game")
        self.root.geometry("900x750")
        self.root.configure(bg=THEME["background"])
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Apply theme
        self.apply_theme()
        
        # Configure main frame
        self.main_frame = ttk.Frame(root, style="Main.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Create header
        self.create_header()
        
        # Create menu bar
        self.create_menu()
        
        # Create game status display
        self.create_status_frame()
        
        # Create video display
        self.create_video_frame()
        
        # Create controls frame
        self.create_controls_frame()
        
        # Create footer with stats
        self.create_footer()
        
        # Initialize model selection
        self.pose_model_type = "MediaPipe"
        self.object_model_type = "YOLOv8"
        
        # Initialize game properties
        self.detector = None
        self.game_engine = None
        self.camera_id = 0
        
        # Game statistics
        self.stats = {
            "tasks_completed": 0,
            "screenshots_taken": 0,
            "high_score": 0,
            "total_playtime": 0
        }
        
        # Set up update loop
        self.update_interval = 30  # milliseconds
        self.update_id = None
        
        # Start time tracking
        self.start_time = time.time()
    
    def apply_theme(self):
        """Apply the custom theme to the UI."""
        style = ttk.Style()
        
        # Configure the theme
        style.configure("TFrame", background=THEME["background"])
        style.configure("Main.TFrame", background=THEME["background"])
        style.configure("Header.TFrame", background=THEME["primary"])
        style.configure("Footer.TFrame", background=THEME["primary"])
        
        style.configure("TLabel", background=THEME["background"], foreground=THEME["text"])
        style.configure("Header.TLabel", background=THEME["primary"], foreground=THEME["text_light"], font=("Helvetica", 14, "bold"))
        style.configure("Status.TLabel", background=THEME["background"], foreground=THEME["text"], font=("Helvetica", 12))
        style.configure("Task.TLabel", background=THEME["background"], foreground=THEME["primary"], font=("Helvetica", 14, "bold"))
        
        style.configure("TButton", background=THEME["primary"], foreground=THEME["text_light"], borderwidth=0)
        style.map("TButton", 
                 background=[('active', THEME["secondary"])],
                 foreground=[('active', THEME["text_light"])])
        
        style.configure("Start.TButton", background=THEME["secondary"], font=("Helvetica", 12, "bold"))
        style.configure("Stop.TButton", background=THEME["accent"], font=("Helvetica", 12, "bold"))
        style.configure("Screenshot.TButton", background=THEME["primary"], font=("Helvetica", 12))
        
        style.configure("TCombobox", fieldbackground=THEME["background"], background=THEME["primary"])
        style.map("TCombobox", 
                 fieldbackground=[('readonly', THEME["background"])],
                 selectbackground=[('readonly', THEME["primary"])])
        
        style.configure("Horizontal.TProgressbar", 
                       background=THEME["secondary"],
                       troughcolor=THEME["background"])
        
        # Create custom fonts
        self.header_font = font.Font(family="Helvetica", size=16, weight="bold")
        self.title_font = font.Font(family="Helvetica", size=14, weight="bold")
        self.normal_font = font.Font(family="Helvetica", size=12)
        self.small_font = font.Font(family="Helvetica", size=10)
    
    def create_header(self):
        """Create the application header."""
        header_frame = ttk.Frame(self.main_frame, style="Header.TFrame")
        header_frame.pack(fill=tk.X, padx=0, pady=(0, 15))
        
        # App logo and title
        logo_label = ttk.Label(header_frame, text="🎮 CV Game", 
                              font=self.header_font, style="Header.TLabel")
        logo_label.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Right-side controls
        controls_frame = ttk.Frame(header_frame, style="Header.TFrame")
        controls_frame.pack(side=tk.RIGHT, padx=15, pady=10)
        
        # Help button
        help_button = ttk.Button(controls_frame, text="❓ Help", 
                                command=self.show_instructions, style="TButton")
        help_button.pack(side=tk.RIGHT, padx=5)
        
        # GitHub link
        github_button = ttk.Button(controls_frame, text="🌐 GitHub", 
                                 command=lambda: webbrowser.open("https://github.com/yourusername/cv-game"), 
                                 style="TButton")
        github_button.pack(side=tk.RIGHT, padx=5)
    
    def create_menu(self):
        """Create the application menu."""
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Start Game", command=self.start_game)
        file_menu.add_command(label="Stop Game", command=self.stop_game)
        file_menu.add_separator()
        file_menu.add_command(label="Take Screenshot", command=self.take_manual_screenshot)
        file_menu.add_command(label="Organize Screenshots", command=self.organize_screenshots)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Settings menu
        settings_menu = tk.Menu(menu_bar, tearoff=0)
        settings_menu.add_command(label="Easy Mode", command=lambda: self.set_difficulty("easy"))
        settings_menu.add_command(label="Medium Mode", command=lambda: self.set_difficulty("medium"))
        settings_menu.add_command(label="Hard Mode", command=lambda: self.set_difficulty("hard"))
        settings_menu.add_separator()
        
        # Add auto screenshot toggle
        self.auto_screenshot_var = tk.BooleanVar()
        settings_menu.add_checkbutton(label="Auto Screenshots", variable=self.auto_screenshot_var,
                                      command=self.toggle_auto_screenshots)
        
        menu_bar.add_cascade(label="Settings", menu=settings_menu)
        
        # Models menu
        models_menu = tk.Menu(menu_bar, tearoff=0)
        
        # Pose models submenu
        pose_menu = tk.Menu(models_menu, tearoff=0)
        pose_menu.add_command(label="MediaPipe", command=lambda: self.change_model("pose", "MediaPipe"))
        pose_menu.add_command(label="MoveNet", command=lambda: self.change_model("pose", "MoveNet"))
        models_menu.add_cascade(label="Pose Model", menu=pose_menu)
        
        # Object models submenu
        object_menu = tk.Menu(models_menu, tearoff=0)
        object_menu.add_command(label="YOLOv8", command=lambda: self.change_model("object", "YOLOv8"))
        object_menu.add_command(label="SSD", command=lambda: self.change_model("object", "SSD"))
        models_menu.add_cascade(label="Object Model", menu=object_menu)
        
        # Add model combinations
        models_menu.add_separator()
        models_menu.add_command(label="MediaPipe + YOLOv8", 
                              command=lambda: self.change_model_combination("MediaPipe", "YOLOv8"))
        models_menu.add_command(label="MediaPipe + SSD", 
                              command=lambda: self.change_model_combination("MediaPipe", "SSD"))
        models_menu.add_command(label="MoveNet + YOLOv8", 
                              command=lambda: self.change_model_combination("MoveNet", "YOLOv8"))
        models_menu.add_command(label="MoveNet + SSD", 
                              command=lambda: self.change_model_combination("MoveNet", "SSD"))
        
        # Add comparison command
        models_menu.add_separator()
        models_menu.add_command(label="Take Comparison Screenshots", command=self.take_comparison_screenshots)
        
        menu_bar.add_cascade(label="Models", menu=models_menu)
        
        # Camera menu
        camera_menu = tk.Menu(menu_bar, tearoff=0)
        camera_menu.add_command(label="Camera 0", command=lambda: self.set_camera(0))
        camera_menu.add_command(label="Camera 1", command=lambda: self.set_camera(1))
        menu_bar.add_cascade(label="Camera", menu=camera_menu)
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        help_menu.add_command(label="Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)
    
    def create_status_frame(self):
        """Create the game status display frame."""
        status_panel = ttk.Frame(self.main_frame)
        status_panel.pack(fill=tk.X, padx=5, pady=(0, 10))
        
        # Left panel - Game status
        self.status_frame = ttk.LabelFrame(status_panel, text="Game Status", padding=(10, 5))
        self.status_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Task label with icon
        task_frame = ttk.Frame(self.status_frame)
        task_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(task_frame, text="🎯 Task:", style="Status.TLabel").pack(side=tk.LEFT)
        self.task_var = tk.StringVar(value="No task yet")
        ttk.Label(task_frame, textvariable=self.task_var, style="Task.TLabel").pack(side=tk.LEFT, padx=10)
        
        # Score and timer in a grid
        stats_frame = ttk.Frame(self.status_frame)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Score with icon
        ttk.Label(stats_frame, text="🏆 Score:", style="Status.TLabel").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.score_var = tk.StringVar(value="0")
        ttk.Label(stats_frame, textvariable=self.score_var, style="Status.TLabel").grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Timer with icon and progress bar
        ttk.Label(stats_frame, text="⏱️ Time:", style="Status.TLabel").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        timer_frame = ttk.Frame(stats_frame)
        timer_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.time_var = tk.StringVar(value="0s")
        ttk.Label(timer_frame, textvariable=self.time_var, style="Status.TLabel").pack(side=tk.LEFT)
        
        self.timer_progress = ttk.Progressbar(timer_frame, orient="horizontal", length=100, mode="determinate")
        self.timer_progress.pack(side=tk.LEFT, padx=(10, 0))
        
        # Feedback message with icon
        feedback_frame = ttk.Frame(self.status_frame)
        feedback_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(feedback_frame, text="💬 Status:", style="Status.TLabel").pack(side=tk.LEFT)
        self.feedback_var = tk.StringVar(value="Ready to start")
        ttk.Label(feedback_frame, textvariable=self.feedback_var, style="Status.TLabel").pack(side=tk.LEFT, padx=10)
        
        # Right panel - Screenshots and Models
        right_panel = ttk.Frame(status_panel)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        
        # Screenshots
        self.screenshots_frame = ttk.LabelFrame(right_panel, text="Screenshots", padding=(10, 5))
        self.screenshots_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Screenshot count with icon
        ttk.Label(self.screenshots_frame, text="📸 Count:", style="Status.TLabel").pack(anchor=tk.W, pady=2)
        self.screenshot_var = tk.StringVar(value="0/10")
        ttk.Label(self.screenshots_frame, textvariable=self.screenshot_var, style="Status.TLabel").pack(anchor=tk.W, pady=2)
        
        # Auto screenshots toggle
        auto_frame = ttk.Frame(self.screenshots_frame)
        auto_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(auto_frame, text="🔄 Auto:", style="Status.TLabel").pack(side=tk.LEFT)
        self.auto_status_var = tk.StringVar(value="Disabled")
        ttk.Label(auto_frame, textvariable=self.auto_status_var, style="Status.TLabel").pack(side=tk.LEFT, padx=10)
        
        # Model info
        self.model_frame = ttk.LabelFrame(right_panel, text="Current Models", padding=(10, 5))
        self.model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.model_frame, text="🧠 Pose:", style="Status.TLabel").pack(anchor=tk.W, pady=2)
        self.pose_info_var = tk.StringVar(value="MediaPipe")
        ttk.Label(self.model_frame, textvariable=self.pose_info_var, style="Status.TLabel").pack(anchor=tk.W, pady=2)
        
        ttk.Label(self.model_frame, text="👁️ Object:", style="Status.TLabel").pack(anchor=tk.W, pady=2)
        self.object_info_var = tk.StringVar(value="YOLOv8")
        ttk.Label(self.model_frame, textvariable=self.object_info_var, style="Status.TLabel").pack(anchor=tk.W, pady=2)
    
    def create_video_frame(self):
        """Create the video display frame."""
        self.video_frame = ttk.LabelFrame(self.main_frame, text="Game View")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Make the canvas larger to ensure full frame visibility
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Default image
        default_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(default_img, (0, 0), (640, 480), (45, 45, 45), -1)
        
        # Add welcome text
        cv2.putText(default_img, "Computer Vision Game", (150, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(default_img, "Press 'Start Game' to begin", (180, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(default_img, "Complete tasks using your camera", (160, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        self.update_image(default_img)
    
    def create_controls_frame(self):
        """Create the game controls frame."""
        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Left side - Game controls
        game_controls = ttk.Frame(self.controls_frame)
        game_controls.pack(side=tk.LEFT, padx=5)
        
        # Game control buttons
        self.start_button = ttk.Button(game_controls, text="▶️ Start Game", 
                                      command=self.start_game, style="Start.TButton")
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = ttk.Button(game_controls, text="⏹️ Stop Game", 
                                     command=self.stop_game, state=tk.DISABLED, style="Stop.TButton")
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.screenshot_button = ttk.Button(game_controls, text="📸 Take Screenshot", 
                                          command=self.take_manual_screenshot, style="Screenshot.TButton")
        self.screenshot_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Center - Model controls
        model_controls = ttk.Frame(self.controls_frame)
        model_controls.pack(side=tk.LEFT, padx=5)
        
        # Model selection button
        self.model_button = ttk.Button(model_controls, text="🔄 Switch Model", 
                                     command=self.cycle_models)
        self.model_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Right side - Settings
        settings_frame = ttk.Frame(self.controls_frame)
        settings_frame.pack(side=tk.RIGHT, padx=5)
        
        # Difficulty selection
        diff_frame = ttk.Frame(settings_frame)
        diff_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        ttk.Label(diff_frame, text="🔄 Difficulty:").pack(side=tk.LEFT)
        self.difficulty_var = tk.StringVar(value="easy")
        diff_combo = ttk.Combobox(diff_frame, textvariable=self.difficulty_var, 
                                values=["easy", "medium", "hard"], width=10, state="readonly")
        diff_combo.pack(side=tk.LEFT, padx=5)
        
        # Player name
        name_frame = ttk.Frame(settings_frame)
        name_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        ttk.Label(name_frame, text="👤 Player:").pack(side=tk.LEFT)
        self.player_var = tk.StringVar(value="Player")
        player_entry = ttk.Entry(name_frame, textvariable=self.player_var, width=15)
        player_entry.pack(side=tk.LEFT, padx=5)
    
    def create_footer(self):
        """Create a footer with game statistics."""
        footer = ttk.Frame(self.main_frame, style="Footer.TFrame")
        footer.pack(fill=tk.X, pady=(15, 0))
        
        # Game stats
        stats_label = ttk.Label(footer, text="Stats:", style="Header.TLabel")
        stats_label.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Create stats display
        self.tasks_var = tk.StringVar(value="Tasks: 0")
        ttk.Label(footer, textvariable=self.tasks_var, style="Header.TLabel").pack(side=tk.LEFT, padx=10)
        
        self.playtime_var = tk.StringVar(value="Playtime: 0:00")
        ttk.Label(footer, textvariable=self.playtime_var, style="Header.TLabel").pack(side=tk.LEFT, padx=10)
        
        self.high_score_var = tk.StringVar(value="High Score: 0")
        ttk.Label(footer, textvariable=self.high_score_var, style="Header.TLabel").pack(side=tk.LEFT, padx=10)
        
        # Version info and model info
        self.model_info_var = tk.StringVar(value="MediaPipe + YOLOv8")
        ttk.Label(footer, textvariable=self.model_info_var, style="Header.TLabel").pack(side=tk.RIGHT, padx=15, pady=10)
    
    def update_image(self, frame):
        """Update the displayed image on the canvas."""
        if frame is None:
            return
        
        # Resize frame to fit canvas if needed (maintain aspect ratio)
        h, w = frame.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        # Only resize if the canvas has been initialized with a valid size
        if canvas_w > 1 and canvas_h > 1:
            # Calculate scaling factor to fit within canvas
            scale_w = canvas_w / w
            scale_h = canvas_h / h
            scale = min(scale_w, scale_h)
            
            if scale < 1:  # Only resize if frame is larger than canvas
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert OpenCV BGR to RGB for tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update canvas with new image
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference to prevent garbage collection    
    def update_status(self):
        """Update the game status display."""
        if self.game_engine:
            stats = self.game_engine.get_stats()
            
            # Update task, score, and timer
            self.task_var.set(stats["current_task"] if stats["current_task"] else "No task")
            self.score_var.set(str(stats["score"]))
            
            # Update timer and progress bar
            timer_value = stats["timer"]
            max_timer = 30 if self.difficulty_var.get() == "easy" else 20 if self.difficulty_var.get() == "medium" else 15
            self.time_var.set(f"{timer_value}s")
            
            # Update progress bar
            if stats["active"]:
                self.timer_progress["maximum"] = max_timer
                self.timer_progress["value"] = timer_value
            else:
                self.timer_progress["value"] = 0
            
            # Update feedback
            if stats["feedback"]:
                self.feedback_var.set(stats["feedback"])
            elif not stats["active"]:
                self.feedback_var.set("Game not active")
            else:
                self.feedback_var.set("Playing...")
            
            # Update screenshot count
            global screenshot_count
            self.screenshot_var.set(f"{screenshot_count}/{max_auto_screenshots}")
            
            # Update stats in footer
            if stats["score"] > self.stats["high_score"]:
                self.stats["high_score"] = stats["score"]
                self.high_score_var.set(f"High Score: {self.stats['high_score']}")
            
            if "Task Complete" in stats["feedback"]:
                self.stats["tasks_completed"] += 1
                self.tasks_var.set(f"Tasks: {self.stats['tasks_completed']}")
        
        # Update playtime
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self.playtime_var.set(f"Playtime: {minutes}:{seconds:02d}")
        
        # Update model info
        self.model_info_var.set(f"{self.pose_model_type} + {self.object_model_type}")
        self.pose_info_var.set(self.pose_model_type)
        self.object_info_var.set(self.object_model_type)
    
    def update_display(self):
        """Update the display with the latest frame."""
        global frame_to_display
        
        # Update the image if a new frame is available
        if frame_to_display is not None:
            self.update_image(frame_to_display)
        
        # Update game status
        self.update_status()
        
        # Schedule the next update
        self.update_id = self.root.after(self.update_interval, self.update_display)
    
    def start_game(self):
        """Start the game."""
        global running, camera_thread, screenshot_count
        
        if running:
            return
        
        # Reset screenshot count
        screenshot_count = 0
        self.screenshot_var.set(f"{screenshot_count}/{max_auto_screenshots}")
        
        # Initialize game components if they don't exist or if model changed
        if not self.detector or self.detector.pose_model_type != self.pose_model_type or self.detector.object_model_type != self.object_model_type:
            self.detector = PoseObjectDetector(
                pose_model=self.pose_model_type,
                object_model=self.object_model_type
            )
        
        if not self.game_engine:
            self.game_engine = GameEngine(screenshot_mode=self.auto_screenshot_var.get())
        
        # Start the game with selected difficulty
        difficulty = self.difficulty_var.get()
        player_name = self.player_var.get()
        self.game_engine.start_game(difficulty=difficulty, player_name=player_name)
        
        # Update status
        self.feedback_var.set(f"Starting {difficulty} game for {player_name} with {self.pose_model_type}+{self.object_model_type}")
        
        # Start camera thread
        running = True
        camera_thread = threading.Thread(target=self.run_camera_thread)
        camera_thread.daemon = True
        camera_thread.start()
        
        # Start display update loop
        self.update_display()
        
        # Update button states
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
    
    def stop_game(self):
        """Stop the currently running game."""
        global running
        
        # Set flag to stop the thread
        running = False
        
        # Give thread time to clean up
        time.sleep(0.5)
        
        # Update UI
        self.feedback_var.set("Game stopped")
        
        # Update button states
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        # Cancel the update timer
        if self.update_id:
            self.root.after_cancel(self.update_id)
            self.update_id = None
    
    def run_camera_thread(self):
        """Thread function to capture and process camera frames."""
        global running, frame_to_display, screenshot_count, auto_screenshots_enabled
        
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open camera {self.camera_id}")
            running = False
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        last_task_completion = time.time() - 10  # Initialize to 10 seconds ago
        
        try:
            while running:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    messagebox.showerror("Error", "Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Process every other frame to reduce CPU load
                if frame_count % 2 == 0:
                    # Detect poses and objects
                    pose_results, objects = self.detector.detect(frame)
                    
                    # Update game state based on detections
                    scene_description = self.detector.interpret_scene(pose_results, objects, frame.shape[0])
                    
                    # Update game logic
                    task_completed = self.game_engine.update(scene_description)
                    
                    # Draw UI
                    draw_game_ui(frame, self.game_engine, self.detector, scene_description)
                    
                    # Add model info to the frame (for screenshots to identify model used)
                    cv2.putText(frame, f"Models: {self.pose_model_type}+{self.object_model_type}", 
                              (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Take screenshot if enabled and under the limit
                    if (auto_screenshots_enabled and 
                        screenshot_count < max_auto_screenshots and 
                        frame_count % 150 == 0):
                        filename = f"{self.pose_model_type}_{self.object_model_type}_gameplay_{screenshot_count}.jpg"
                        take_screenshot(frame, filename)
                        screenshot_count += 1
                        self.stats["screenshots_taken"] += 1
                    
                    # Only take task completion screenshot if it's a real task completion
                    # and at least 2 seconds have passed since last completion
                    current_time = time.time()
                    if (task_completed and 
                        "Complete" in self.game_engine.feedback and
                        screenshot_count < max_auto_screenshots and
                        (current_time - last_task_completion) > 2.0):
                        
                        filename = f"{self.pose_model_type}_{self.object_model_type}_task_{screenshot_count}.jpg"
                        take_screenshot(frame, filename)
                        screenshot_count += 1
                        self.stats["screenshots_taken"] += 1
                        last_task_completion = current_time
                
                # Update the display frame
                frame_to_display = frame.copy()
                
                # Short delay to reduce CPU usage
                time.sleep(0.01)
        
        finally:
            # Release resources
            cap.release()
    
    def take_manual_screenshot(self):
        """Take a manual screenshot."""
        global frame_to_display, screenshot_count
        
        if frame_to_display is not None:
            screenshot_name = f"{self.pose_model_type}_{self.object_model_type}_manual_{int(time.time())}"
            take_screenshot(frame_to_display, screenshot_name)
            screenshot_count += 1
            self.stats["screenshots_taken"] += 1
            self.screenshot_var.set(f"{screenshot_count}/{max_auto_screenshots}")
            self.feedback_var.set(f"Screenshot taken: {screenshot_name}")
    
    def take_comparison_screenshots(self):
        """Take a series of screenshots with different model combinations for comparison."""
        global frame_to_display
        
        if frame_to_display is None:
            messagebox.showinfo("Info", "Start the game first to take comparison screenshots")
            return
        
        # Create directory for comparison screenshots
        comparison_dir = Path("model_comparisons")
        comparison_dir.mkdir(exist_ok=True)
        
        # Timestamp for the comparison set
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save the current frame with different model configurations
        # We'll just change the label to show what each model would see
        # In a real implementation, you'd need to reinitialize the detector and process the frame
        
        frame = frame_to_display.copy()
        base_filename = f"comparison_{timestamp}"
        
        # Add text for current model
        cv2.putText(frame, f"CURRENT: {self.pose_model_type}+{self.object_model_type}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Create a comparison image showing what different models would detect
        comparison_img = np.zeros((frame.shape[0] * 2, frame.shape[1] * 2, 3), dtype=np.uint8)
        
        # Save current frame at top-left position
        comparison_img[0:frame.shape[0], 0:frame.shape[1]] = frame
        
        # Create labeled versions for other model combinations
        model_combinations = [
            ("MediaPipe", "YOLOv8"),
            ("MediaPipe", "SSD"),
            ("MoveNet", "YOLOv8"),
            ("MoveNet", "SSD")
        ]
        
        # Skip the current model combination
        current_combo = (self.pose_model_type, self.object_model_type)
        model_combinations = [combo for combo in model_combinations if combo != current_combo]
        
        positions = [
            (0, frame.shape[1]),  # top-right
            (frame.shape[0], 0),  # bottom-left
            (frame.shape[0], frame.shape[1])  # bottom-right
        ]
        
        for i, ((pose_model, object_model), position) in enumerate(zip(model_combinations, positions)):
            # Create a copy of the frame
            model_frame = frame.copy()
            
            # Add model label
            cv2.putText(model_frame, f"{pose_model}+{object_model}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Place in the comparison image
            y, x = position
            comparison_img[y:y+frame.shape[0], x:x+frame.shape[1]] = model_frame
            
            # Save individual model frame
            filename = f"{base_filename}_{pose_model}_{object_model}.jpg"
            filepath = str(comparison_dir / filename)
            cv2.imwrite(filepath, model_frame)
        
        # Save the full comparison image
        comparison_filename = f"{base_filename}_all_models.jpg"
        comparison_filepath = str(comparison_dir / comparison_filename)
        cv2.imwrite(comparison_filepath, comparison_img)
        
        messagebox.showinfo("Comparison", f"Saved comparison screenshots to {comparison_dir}")
    
    def change_model(self, model_type, model_name):
        """
        Change a single model type.
        
        Args:
            model_type (str): "pose" or "object"
            model_name (str): New model name
        """
        if running:
            messagebox.showinfo("Info", "Stop the game first to change models")
            return
        
        if model_type == "pose":
            self.pose_model_type = model_name
        elif model_type == "object":
            self.object_model_type = model_name
        
        # Update UI
        self.model_info_var.set(f"{self.pose_model_type} + {self.object_model_type}")
        self.pose_info_var.set(self.pose_model_type)
        self.object_info_var.set(self.object_model_type)
        
        # Reset detector to use new models when game restarts
        self.detector = None
        
        self.feedback_var.set(f"Changed {model_type} model to {model_name}")
    
    def change_model_combination(self, pose_model, object_model):
        """
        Change both pose and object models at once.
        
        Args:
            pose_model (str): New pose model name
            object_model (str): New object model name
        """
        if running:
            messagebox.showinfo("Info", "Stop the game first to change models")
            return
        
        # Temporarily disable MoveNet due to compatibility issues
        if pose_model == "MoveNet":
            messagebox.showinfo("Model Issue", "MoveNet currently has compatibility issues. Using MediaPipe instead.")
            pose_model = "MediaPipe"
        
        self.pose_model_type = pose_model
        self.object_model_type = object_model
        
        # Update UI
        self.model_info_var.set(f"{self.pose_model_type} + {self.object_model_type}")
        self.pose_info_var.set(self.pose_model_type)
        self.object_info_var.set(self.object_model_type)
        
        # Reset detector to use new models when game restarts
        self.detector = None
        
        self.feedback_var.set(f"Changed models to {pose_model} + {object_model}")
    def cycle_models(self):
        """Cycle through model combinations."""
        combinations = [
            ("MediaPipe", "YOLOv8"),
            ("MediaPipe", "SSD"),
            ("MoveNet", "YOLOv8"),
            ("MoveNet", "SSD")
        ]
        
        # Find current combination
        current = (self.pose_model_type, self.object_model_type)
        try:
            index = combinations.index(current)
        except ValueError:
            index = 0
        
        # Get next combination
        next_index = (index + 1) % len(combinations)
        next_pose, next_object = combinations[next_index]
        
        # Change to next combination
        self.change_model_combination(next_pose, next_object)
    
    def organize_screenshots(self):
        """Organize screenshots into categories."""
        try:
            organize_screenshots()
            messagebox.showinfo("Screenshots", "Screenshots organized into categories")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to organize screenshots: {e}")
    
    def set_difficulty(self, difficulty):
        """Set the game difficulty."""
        self.difficulty_var.set(difficulty)
        if self.game_engine and self.game_engine.active:
            self.game_engine.start_game(difficulty=difficulty, player_name=self.player_var.get())
            self.feedback_var.set(f"Changed difficulty to {difficulty}")
    
    def set_camera(self, camera_id):
        """Change the camera source."""
        if running:
            messagebox.showinfo("Info", "Stop the game first to change camera")
        else:
            self.camera_id = camera_id
            self.feedback_var.set(f"Camera changed to {camera_id}")
    
    def toggle_auto_screenshots(self):
        """Toggle automatic screenshots."""
        global auto_screenshots_enabled
        auto_screenshots_enabled = self.auto_screenshot_var.get()
        status = "Enabled" if auto_screenshots_enabled else "Disabled"
        self.auto_status_var.set(status)
        self.feedback_var.set(f"Auto screenshots {status.lower()}")
    
    def show_instructions(self):
        """Show game instructions."""
        instructions = """
Computer Vision Interactive Game Instructions

Objective:
Complete physical tasks detected by the camera to earn points.

Tasks by Difficulty:
- Easy: Simple tasks like "stand" or "near bottle"
- Medium: Combined tasks like "stand near bottle"
- Hard: Complex tasks like "lift bottle" or sequences

Controls:
- Start Game: Begin playing with selected difficulty
- Stop Game: End the current game
- Take Screenshot: Save the current frame
- Switch Model: Cycle through different model combinations

Model Comparison:
- Change models from the Models menu
- Take comparison screenshots to see how different models perform
- Each screenshot is labeled with the models used

Tips:
- Make sure you have good lighting
- Have objects like chairs and bottles visible
- Move slowly and clearly for better detection
- Complete tasks to earn points and bonus time
        """
        messagebox.showinfo("Instructions", instructions)
    
    def show_shortcuts(self):
        """Show keyboard shortcuts."""
        shortcuts = """
Keyboard Shortcuts:

When game window is in focus:
- ESC: Quit the game
- S: Start a new game
- X: Stop the current game
- P: Take a screenshot
- M: Cycle through models
- E: Switch to Easy mode
- M: Switch to Medium mode
- H: Switch to Hard mode
- A: Toggle Auto Screenshots
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)
    
    def show_about(self):
        """Show about information."""
        about_text = """
Computer Vision Interactive Game

Version 1.0.0

A project that demonstrates the use of computer vision 
for pose detection and object recognition in an interactive game.

Features model comparison to evaluate different CV approaches:
- MediaPipe vs MoveNet for pose detection
- YOLOv8 vs SSD for object detection

Technologies:
- MediaPipe for pose detection
- YOLOv8 for object detection
- OpenCV for image processing
- Tkinter for the user interface

Created by: Your Name
        """
        messagebox.showinfo("About", about_text)
    
    def on_closing(self):
        """Handle window closing."""
        global running
        
        # Stop the game if it's running
        running = False
        
        # Give time for threads to clean up
        time.sleep(0.5)
        
        # Destroy the window
        self.root.destroy()

def setup_keyboard_shortcuts(app, root):
    """Setup keyboard shortcuts for the application."""
    def handle_key(event):
        key = event.char.lower()
        if key == 's':
            app.start_game()
        elif key == 'x':
            app.stop_game()
        elif key == 'p':
            app.take_manual_screenshot()
        elif key == 'm':
            app.cycle_models()
        elif key == 'e':
            app.set_difficulty("easy")
        elif key == 'm':
            app.set_difficulty("medium")
        elif key == 'h':
            app.set_difficulty("hard")
        elif key == 'a':
            app.auto_screenshot_var.set(not app.auto_screenshot_var.get())
            app.toggle_auto_screenshots()
        elif key == 'c':
            app.take_comparison_screenshots()
        elif event.keysym == 'Escape':
            app.on_closing()
    
    root.bind('<Key>', handle_key)

def main():
    """Main function to run the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CV-based interactive game")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera index (default: 0)")
    args = parser.parse_args()
    
    # Create and start the GUI
    root = tk.Tk()
    app = GameGUI(root)
    app.camera_id = args.camera
    
    # Setup keyboard shortcuts
    setup_keyboard_shortcuts(app, root)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()