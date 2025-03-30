"""
Game engine for the CV-based interactive game.
Simplified version with easier tasks and more facial expressions.
"""
import time
import random
import csv
import os
from pathlib import Path

class GameEngine:
    """
    Game engine for the CV-based interactive game.
    Manages tasks, scoring, and game state.
    """
    
    def __init__(self, screenshot_mode=False):
        """
        Initialize the game engine.
        
        Args:
            screenshot_mode (bool): Whether to take screenshots automatically
        """
        # Simplified game tasks by difficulty level
        self.tasks = {
            "easy": [
                "stand",
                "squat", 
                "near bottle",
                "near chair",
                "smile",                # New task for facial expression
                "close eyes",           # New task for facial expression
                "near phone",           # New task with phone object
                "show palm"             # New task for hand gesture
            ],
            "medium": [
                "stand near bottle", 
                "squat near chair",
                "stand near chair",
                "squat near bottle",
                "lift phone",           # New task for lifting phone
                "smile near bottle",    # Combined facial expression with object
                "close eyes near chair", # Combined facial expression with object
                "show palm near phone"  # Combined hand gesture with object
            ],
            "hard": [
                "lift bottle", 
                "lift chair",
                "lift phone",           # New lifting task
                "squat then stand",
                "stand then squat",
                "smile then close eyes", # Sequence of facial expressions
                "lift phone then smile", # Sequence with object and expression
                "show palm then close eyes" # Sequence with gesture and expression
            ]
        }
        
        # Game state variables
        self.difficulty = "easy"
        self.score = 0
        self.timer = 0
        self.current_task = None
        self.feedback = ""
        self.active = False
        self.player_name = "Player"
        self.screenshot_mode = screenshot_mode
        
        # For multi-step tasks
        self.sequence_state = {}
        
        # Task completion tracking
        self.last_task_completed_time = 0
        self.task_completion_cooldown = 2.0  # seconds
        
        # Initialize leaderboard
        self.leaderboard_file = "leaderboard.csv"
        self.leaderboard = []
        self.load_leaderboard()
    
    def start_game(self, difficulty="easy", player_name="Player"):
        """
        Start a new game.
        
        Args:
            difficulty (str): Game difficulty ("easy", "medium", or "hard")
            player_name (str): Player name for the leaderboard
        """
        # Set game parameters
        self.difficulty = difficulty
        self.player_name = player_name
        self.active = True
        self.score = 0
        
        # Set timer based on difficulty - longer times for easier play
        self.timer = 30 if difficulty == "easy" else 20 if difficulty == "medium" else 15
        
        # Choose random task
        self.current_task = random.choice(self.tasks[difficulty])
        self.feedback = f"New task: {self.current_task}"
        
        # Reset sequence state
        self.sequence_state = {}
        
        print(f"Game started: {difficulty} mode, task: {self.current_task}")
    
    def update(self, scene_description, delta_time=1/30):
        """
        Update game state based on the scene description.
        
        Args:
            scene_description (str): Description of the current scene
            delta_time (float): Time elapsed since last update
            
        Returns:
            bool: Whether a task was completed in this update
        """
        if not self.active or self.timer <= 0:
            if self.timer <= 0 and self.active:
                self.feedback = f"Game Over! Score: {self.score}"
                self.save_score()
                self.active = False
                print(f"Game over. Final score: {self.score}")
            return False
        
        # Update timer
        self.timer -= delta_time
        
        # Flag to track if a task was completed in this update
        task_completed = False
        
        # Check if task is completed
        if self.current_task:
            # Check for task completion with cooldown prevention
            current_time = time.time()
            if (current_time - self.last_task_completed_time) >= self.task_completion_cooldown:
                completed = self._check_task_completion(scene_description)
                
                if completed:
                    # Award points based on difficulty
                    points = 10 if self.difficulty == "easy" else 15 if self.difficulty == "medium" else 20
                    self.score += points
                    
                    # Provide feedback
                    self.feedback = f"Task Complete! +{points} points"
                    
                    # Choose a new task that's different from the current one
                    new_task = self.current_task
                    while new_task == self.current_task and len(self.tasks[self.difficulty]) > 1:
                        new_task = random.choice(self.tasks[self.difficulty])
                    self.current_task = new_task
                    
                    # Add bonus time
                    bonus_time = 8 if self.difficulty == "easy" else 6 if self.difficulty == "medium" else 5
                    self.timer += bonus_time
                    
                    # Reset sequence state for new task
                    self.sequence_state = {}
                    
                    # Update task completion time
                    self.last_task_completed_time = current_time
                    
                    print(f"Task completed! New task: {self.current_task}, Score: {self.score}")
                    
                    # Set flag to indicate task completion
                    task_completed = True
        
        return task_completed
    
    def _check_task_completion(self, scene_description):
        """
        Check if the current task is completed based on the scene description.
        Simplified to make tasks easier to complete.
        
        Args:
            scene_description (str): Description of the current scene
            
        Returns:
            bool: Whether the task is completed
        """
        # Convert everything to lowercase for comparison
        scene_lower = scene_description.lower()
        task_lower = self.current_task.lower()
        
        # Debug print to help with troubleshooting
        print(f"Checking task: '{task_lower}' against scene: '{scene_lower}'")
        
        # Simplified task checking for special keywords
        # Basic actions
        if task_lower == "stand":
            completed = "standing" in scene_lower
        elif task_lower == "squat":
            completed = "squatting" in scene_lower
        elif task_lower == "smile":
            completed = "smiling" in scene_lower
        elif task_lower == "close eyes":
            completed = "eyes closed" in scene_lower or "closed eyes" in scene_lower
        elif task_lower == "show palm":
            completed = "palm" in scene_lower or "hand open" in scene_lower
        
        # Objects
        elif "near bottle" in task_lower:
            completed = "bottle" in scene_lower
        elif "near chair" in task_lower:
            completed = "chair" in scene_lower
        elif "near phone" in task_lower:
            completed = "phone" in scene_lower or "cell phone" in scene_lower
        
        # Lifting actions
        elif "lift bottle" in task_lower:
            completed = "lifting" in scene_lower and "bottle" in scene_lower
        elif "lift chair" in task_lower:
            completed = "lifting" in scene_lower and "chair" in scene_lower
        elif "lift phone" in task_lower:
            completed = "lifting" in scene_lower and ("phone" in scene_lower or "cell phone" in scene_lower)
        
        # Combined facial expressions with objects
        elif "smile near" in task_lower:
            object_type = task_lower.split("smile near ")[1]
            completed = "smiling" in scene_lower and object_type in scene_lower
        elif "close eyes near" in task_lower:
            object_type = task_lower.split("close eyes near ")[1]
            completed = ("eyes closed" in scene_lower or "closed eyes" in scene_lower) and object_type in scene_lower
        elif "show palm near" in task_lower:
            object_type = task_lower.split("show palm near ")[1]
            completed = ("palm" in scene_lower or "hand open" in scene_lower) and object_type in scene_lower
            
        # Check if this is a sequence task (contains "then")
        elif " then " in task_lower:
            # Split into steps
            steps = task_lower.split(" then ")
            
            # Initialize sequence state if it's empty
            if not self.sequence_state:
                self.sequence_state = {
                    "current_step": 0,
                    "steps": steps,
                    "last_update": time.time()
                }
            
            # Get current step
            current_step = steps[self.sequence_state["current_step"]]
            
            # Check current step (simplified)
            step_completed = False
            
            # Basic actions for sequence steps
            if current_step == "stand":
                step_completed = "standing" in scene_lower
            elif current_step == "squat":
                step_completed = "squatting" in scene_lower
            elif current_step == "smile":
                step_completed = "smiling" in scene_lower
            elif current_step == "close eyes":
                step_completed = "eyes closed" in scene_lower or "closed eyes" in scene_lower
            elif current_step == "show palm":
                step_completed = "palm" in scene_lower or "hand open" in scene_lower
            
            # Lifting actions for sequence steps
            elif "lift" in current_step:
                object_type = current_step.split("lift ")[1]
                step_completed = "lifting" in scene_lower and object_type in scene_lower
            
            if step_completed:
                # If this is the last step, task is complete
                if self.sequence_state["current_step"] == len(steps) - 1:
                    print(f"Sequence task completed: {task_lower}")
                    return True
                
                # Otherwise, move to the next step
                self.sequence_state["current_step"] += 1
                self.sequence_state["last_update"] = time.time()
                
                # Update feedback to show progress
                next_step = steps[self.sequence_state["current_step"]]
                self.feedback = f"Good! Now {next_step}"
                
                print(f"Sequence step {self.sequence_state['current_step']} completed: {current_step}")
                return False
            
            # Check for timeout on steps (15 seconds - longer for easier play)
            if time.time() - self.sequence_state["last_update"] > 15:
                # Reset to first step
                self.sequence_state["current_step"] = 0
                self.sequence_state["last_update"] = time.time()
                self.feedback = f"Timed out. Start over: {steps[0]}"
                
                print(f"Sequence step timed out, resetting to first step")
            
            return False
        else:
            # More flexible task matching - check if all task words are in the scene
            task_words = task_lower.split()
            completed = all(word in scene_lower for word in task_words)
            
        if completed:
            print(f"Task completed: {task_lower}")
        return completed
    
    def load_leaderboard(self):
        """Load the leaderboard from the CSV file."""
        self.leaderboard = []
        
        # Create leaderboard file if it doesn't exist
        if not os.path.exists(self.leaderboard_file):
            self.leaderboard = [("AI Player", 100), ("Bot", 75), ("Computer", 50)]
            self.save_score()
            return
        
        try:
            with open(self.leaderboard_file, "r") as f:
                reader = csv.reader(f)
                # Skip header
                next(reader, None)
                self.leaderboard = [(name, int(score)) for name, score in reader]
        except Exception as e:
            print(f"Error loading leaderboard: {e}")
            self.leaderboard = [("AI Player", 100), ("Bot", 75), ("Computer", 50)]
    
    def save_score(self):
        """Save the current score to the leaderboard."""
        # Add current score if game was played
        if self.score > 0:
            self.leaderboard.append((self.player_name, self.score))
            
            # Sort by score (descending) and keep top 5
            self.leaderboard = sorted(self.leaderboard, key=lambda x: x[1], reverse=True)[:5]
        
        try:
            with open(self.leaderboard_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Score"])
                writer.writerows(self.leaderboard)
        except Exception as e:
            print(f"Error saving leaderboard: {e}")
    
    def get_stats(self):
        """
        Get current game statistics.
        
        Returns:
            dict: Dictionary containing game statistics
        """
        return {
            "active": self.active,
            "difficulty": self.difficulty,
            "score": self.score,
            "timer": max(0, int(self.timer)),
            "current_task": self.current_task,
            "feedback": self.feedback,
            "leaderboard": self.leaderboard[:5]
        }