"""
Screenshot utility for the CV-based game.
Enhanced version with better organization and management.
"""
import cv2
import os
import time
import datetime
from pathlib import Path
import shutil

def take_screenshot(frame, name=None, max_per_session=50):
    """
    Take a screenshot and save it to the screenshots directory.
    
    Args:
        frame: The frame to save
        name: Optional name for the screenshot
        max_per_session: Maximum screenshots per session
        
    Returns:
        str: Path to the saved screenshot or None if limit reached
    """
    # Create screenshots directory if it doesn't exist
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    
    # Check if we've reached the maximum number of screenshots
    existing_screenshots = list(screenshots_dir.glob("*.jpg"))
    if len(existing_screenshots) >= max_per_session:
        # If we have too many screenshots, delete the oldest
        if existing_screenshots:
            oldest_screenshot = min(existing_screenshots, key=os.path.getctime)
            try:
                os.remove(oldest_screenshot)
                print(f"Deleted oldest screenshot: {oldest_screenshot}")
            except Exception as e:
                print(f"Error deleting screenshot: {e}")
    
    # Generate filename
    if name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"screenshot_{timestamp}"
    
    # Ensure filename has no spaces and has .jpg extension
    filename = f"{name.replace(' ', '_')}.jpg"
    filepath = os.path.join("screenshots", filename)
    
    # Save the image with good quality
    try:
        # Add timestamp to the image
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        h, w = frame.shape[:2]
        cv2.putText(frame, timestamp, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save the image
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Screenshot saved to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return None

def organize_screenshots():
    """
    Organize screenshots into categories based on filename.
    Creates subfolders for different types of screenshots.
    """
    screenshots_dir = Path("screenshots")
    
    # Define categories and their corresponding prefixes
    categories = {
        "gameplay": ["gameplay_"],
        "tasks": ["task_complete_"],
        "manual": ["manual_screenshot_"]
    }
    
    # Create category directories
    for category in categories:
        category_dir = screenshots_dir / category
        category_dir.mkdir(exist_ok=True)
    
    # Move files to their categories
    for file in screenshots_dir.glob("*.jpg"):
        filename = file.name
        moved = False
        
        for category, prefixes in categories.items():
            if any(filename.startswith(prefix) for prefix in prefixes):
                dest = screenshots_dir / category / filename
                try:
                    # Don't move files that are already in category folders
                    if file.parent.name == "screenshots" and file.is_file():
                        shutil.copy2(file, dest)
                        os.remove(file)
                    moved = True
                    break
                except Exception as e:
                    print(f"Error moving {filename} to {category}: {e}")
        
        # If file doesn't match any category, move to "other"
        if not moved and file.parent.name == "screenshots" and file.is_file():
            other_dir = screenshots_dir / "other"
            other_dir.mkdir(exist_ok=True)
            try:
                shutil.copy2(file, other_dir / filename)
                os.remove(file)
            except Exception as e:
                print(f"Error moving {filename} to other: {e}")

def create_montage(title="Game Highlights"):
    """
    Create a montage of screenshots as a single image.
    
    Args:
        title: Title for the montage
    Returns:
        str: Path to the saved montage image
    """
    import numpy as np
    screenshots_dir = Path("screenshots")
    # Get list of all JPG files
    screenshots = []
    for category in ["tasks", "gameplay", "manual"]:
        category_dir = screenshots_dir / category
        if category_dir.exists():
            screenshots.extend(list(category_dir.glob("*.jpg")))
    # Add any remaining JPGs in the main directory
    screenshots.extend(list(screenshots_dir.glob("*.jpg")))
    if not screenshots:
        print("No screenshots found to create montage")
        return None
    # Sort by creation time (newest first)
    screenshots = sorted(screenshots, key=os.path.getctime, reverse=True)
    # Limit to most recent 9 screenshots
    screenshots = screenshots[:9]
    # Read images
    images = []
    for scr in screenshots:
        img = cv2.imread(str(scr))
        if img is not None:
            # Resize to a consistent size
            img = cv2.resize(img, (320, 240))
            images.append(img)
    if not images:
        print("No valid images found for montage")
        return None
    # Create a 3x3 grid (or smaller if fewer images)
    rows = min(3, (len(images) + 2) // 3)
    cols = min(3, len(images))
    # Create empty montage canvas
    montage = np.zeros((rows * 240, cols * 320, 3), dtype=np.uint8)
    # Place images in grid
    for i, img in enumerate(images):
        r, c = i // cols, i % cols
        montage[r*240:(r+1)*240, c*320:(c+1)*320] = img
    # Add title
    cv2.putText(montage, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Add timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(montage, timestamp, (10, montage.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Save montage
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    montage_path = str(screenshots_dir / f"montage_{timestamp_str}.jpg")
    cv2.imwrite(montage_path, montage, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Created montage at {montage_path}")
    return montage_path

def get_screenshot_stats():
    """
    Get statistics about saved screenshots.
    
    Returns:
        dict: Dictionary containing screenshot statistics
    """
    screenshots_dir = Path("screenshots")
    
    stats = {
        "total": 0,
        "categories": {},
        "newest": None,
        "oldest": None
    }
    
    # Check if directory exists
    if not screenshots_dir.exists():
        return stats
    
    # Count total screenshots
    all_screenshots = list(screenshots_dir.glob("**/*.jpg"))
    stats["total"] = len(all_screenshots)
    
    # Count by category
    for category in ["gameplay", "tasks", "manual", "other"]:
        category_dir = screenshots_dir / category
        if category_dir.exists():
            count = len(list(category_dir.glob("*.jpg")))
            stats["categories"][category] = count
    
    # Find newest and oldest
    if all_screenshots:
        newest = max(all_screenshots, key=os.path.getctime)
        oldest = min(all_screenshots, key=os.path.getctime)
        
        stats["newest"] = {
            "path": str(newest),
            "time": datetime.datetime.fromtimestamp(os.path.getctime(newest)).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        stats["oldest"] = {
            "path": str(oldest),
            "time": datetime.datetime.fromtimestamp(os.path.getctime(oldest)).strftime("%Y-%m-%d %H:%M:%S")
        }
    
    return stats