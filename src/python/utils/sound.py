"""
Simple sound notification utility for training and execution scripts.
Uses system commands to play sounds.
"""

import os
import subprocess
import sys


def play_sound(sound_type="default"):
    """
    Play a system sound notification.
    
    Args:
        sound_type: Type of sound to play
            - "default": Generic completion sound
            - "success": Success/training complete sound
            - "alert": Alert/notification sound
            - "error": Error sound
    """
    try:
        if sys.platform == "darwin":  # macOS
            if sound_type == "success":
                os.system('afplay /System/Library/Sounds/Glass.aiff')
            elif sound_type == "alert":
                os.system('afplay /System/Library/Sounds/Ping.aiff')
            elif sound_type == "error":
                os.system('afplay /System/Library/Sounds/Basso.aiff')
            else:
                os.system('afplay /System/Library/Sounds/Purr.aiff')
                
        elif sys.platform == "linux":
            # Try various sound playing methods
            if sound_type == "success":
                sound_cmd = "paplay /usr/share/sounds/freedesktop/stereo/complete.oga 2>/dev/null || echo -e '\\a'"
            elif sound_type == "alert":
                sound_cmd = "paplay /usr/share/sounds/freedesktop/stereo/message.oga 2>/dev/null || echo -e '\\a'"
            elif sound_type == "error":
                sound_cmd = "paplay /usr/share/sounds/freedesktop/stereo/dialog-error.oga 2>/dev/null || echo -e '\\a'"
            else:
                sound_cmd = "paplay /usr/share/sounds/freedesktop/stereo/bell.oga 2>/dev/null || echo -e '\\a'"
            
            # Try to play sound
            subprocess.run(sound_cmd, shell=True, check=False)
            
        elif sys.platform == "win32":  # Windows
            import winsound
            if sound_type == "success":
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            elif sound_type == "alert":
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            elif sound_type == "error":
                winsound.MessageBeep(winsound.MB_ICONHAND)
            else:
                winsound.MessageBeep(winsound.MB_OK)
                
    except Exception:
        # Fallback: just print a message
        print(f"[Sound] {sound_type} notification would play here")
        # Try simple bell character as last resort
        print("\a", end="", flush=True)
