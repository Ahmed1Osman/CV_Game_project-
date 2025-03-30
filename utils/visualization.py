"""
Visualization utilities for the CV-based game.
Handles UI drawing and visual elements.
"""
import cv2
import numpy as np

def draw_game_ui(frame, game_engine, detector, scene_description):
    """
    Draw the game UI on the frame.
    
    Args:
        frame: The frame to draw on
        game_engine: The game engine instance
        detector: The pose/object detector
        scene_description: Current scene description
    """
    h, w = frame.shape[:2]
    
    # Draw semi-transparent overlay for better text visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h-110), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Draw header with game info
    cv2.putText(frame, "CV Game", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, f"Action: {detector.current_action}", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Draw scene description
    draw_text_with_bg(frame, f"Scene: {scene_description}", (10, h-80), 0.7, (0, 0, 0), (0, 255, 0))
    
    # Draw game info if active
    if game_engine.active:
        # Draw task
        task_color = (255, 200, 0)  # Yellow
        draw_text_with_bg(frame, f"Task: {game_engine.current_task}", (10, 60), 0.7, (0, 0, 0), task_color)
        
        # Draw timer - change color when time is low
        timer_color = (0, 255, 0) if game_engine.timer > 5 else (0, 165, 255) if game_engine.timer > 3 else (0, 0, 255)
        cv2.putText(frame, f"Time: {int(game_engine.timer)}s", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)
        
        # Draw score
        cv2.putText(frame, f"Score: {game_engine.score}", (w-150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw feedback message
        if game_engine.feedback:
            bg_color = (0, 255, 0) if "Complete" in game_engine.feedback else (0, 0, 255)
            draw_text_with_bg(frame, game_engine.feedback, (int(w/2)-150, h-25), 0.8, (0, 0, 0), bg_color)
    else:
        # Draw game over message or instructions
        if game_engine.score > 0:
            cv2.putText(frame, f"Game Over! Score: {game_engine.score}", (int(w/2)-150, h-25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press 'Start Game' to begin", (int(w/2)-150, h-25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def draw_text_with_bg(img, text, pos, font_scale, text_color, bg_color, thickness=2, padding=5):
    """
    Draw text with a colored background rectangle.
    
    Args:
        img: Image to draw on
        text: Text string
        pos: Position (x, y)
        font_scale: Font scale
        text_color: Text color (BGR)
        bg_color: Background color (BGR)
        thickness: Text thickness
        padding: Padding around text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Draw background rectangle
    cv2.rectangle(
        img, 
        (pos[0]-padding, pos[1]-text_size[1]-padding), 
        (pos[0]+text_size[0]+padding, pos[1]+padding), 
        bg_color, 
        -1
    )
    
    # Draw text
    cv2.putText(
        img, 
        text, 
        pos, 
        font, 
        font_scale, 
        text_color, 
        thickness
    )