"""
Wine Key Sender
---------------
Launches an executable inside a specific Wine environment/prefix
and sends key combinations to it with a configurable interval.

Requirements:
    pip install pyautogui xdotool
    (xdotool must also be installed on the system: sudo apt install xdotool)
"""

import subprocess
import time
import os
import sys

# ──────────────────────────────────────────────
# CONFIGURATION — edit these values
# ──────────────────────────────────────────────

WINE_PREFIX   = "/home/yordano/.mt5"             # Path to the Wine prefix (WINEPREFIX)
WINE_BIN      = "wine"                           # Wine binary; use "wine64" if needed
EXE_PATH      = "C:\\Program Files\\MetaTrader 5\\MetaEditor64.exe"  # No stray quote!
INTERVAL      = 2.0                              # Seconds between each keystroke/combo

# List of key combinations to send, in order.
# Use xdotool key names: https://gitlab.com/cunidev/gestures/-/wikis/xdotool-list-of-key-codes
# Single key  → "Return"
# With mods   → "ctrl+s", "alt+F4", "shift+Tab"
KEY_COMBOS = [
    "alt+f",       # File menu
    "1",           # Load file (requires that user has already opened the desired .mq5 file at least once)
    "F7",          # Compile
    "ctrl+w",      # Close file
    "alt+F4",      # Close window
]

STARTUP_WAIT  = 6.0    # Seconds to wait for the app to fully load (MetaEditor can be slow)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def launch_wine_app() -> subprocess.Popen:
    """Launch the .exe inside the specified Wine prefix."""
    env = os.environ.copy()
    env["WINEPREFIX"] = WINE_PREFIX

    print(f"[*] Launching: {WINE_BIN} {EXE_PATH}")
    print(f"[*] Using WINEPREFIX: {WINE_PREFIX}")

    proc = subprocess.Popen(
        [WINE_BIN, EXE_PATH],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[*] Process started (PID: {proc.pid})")
    return proc


def list_all_windows() -> None:
    """Print all visible window IDs and their titles — useful for debugging."""
    result = subprocess.run(
        ["xdotool", "search", "--onlyvisible", "--name", ""],
        capture_output=True, text=True
    )
    ids = result.stdout.strip().splitlines()
    if not ids:
        print("  [debug] No visible windows found by xdotool.")
        return
    print("  [debug] Currently visible windows:")
    for wid in ids:
        name_res = subprocess.run(
            ["xdotool", "getwindowname", wid],
            capture_output=True, text=True
        )
        print(f"    ID {wid}: {name_res.stdout.strip()}")


def get_window_id(
    title_fragment: str,
    proc: subprocess.Popen | None = None,
    retries: int = 20,
    delay: float = 2.0,
) -> str | None:
    """
    Use xdotool to find the window ID of the launched app.
    Tries two strategies:
      1. Match by window title fragment (case-insensitive substring via --name).
      2. Match by PID if `proc` is provided (Wine spawns child processes, so
         we search the whole process tree).
    Polls up to `retries` times with `delay` seconds between attempts.
    """
    for attempt in range(1, retries + 1):
        # Strategy 1 — title match
        result = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--name", title_fragment],
            capture_output=True, text=True
        )
        ids = result.stdout.strip().splitlines()
        if ids:
            wid = ids[-1]
            name_res = subprocess.run(
                ["xdotool", "getwindowname", wid],
                capture_output=True, text=True
            )
            print(f"[*] Window found by title (ID: {wid}, name: '{name_res.stdout.strip()}') on attempt {attempt}")
            return wid

        # Strategy 2 — PID-based search (walks child pids of the wine launcher)
        if proc is not None:
            try:
                # Get all descendant PIDs via pgrep
                pgrep = subprocess.run(
                    ["pgrep", "-P", str(proc.pid)],
                    capture_output=True, text=True
                )
                child_pids = pgrep.stdout.strip().splitlines()
                all_pids = [str(proc.pid)] + child_pids

                for pid in all_pids:
                    pid_result = subprocess.run(
                        ["xdotool", "search", "--onlyvisible", "--pid", pid],
                        capture_output=True, text=True
                    )
                    pid_ids = pid_result.stdout.strip().splitlines()
                    if pid_ids:
                        wid = pid_ids[-1]
                        name_res = subprocess.run(
                            ["xdotool", "getwindowname", wid],
                            capture_output=True, text=True
                        )
                        print(f"[*] Window found by PID {pid} (ID: {wid}, name: '{name_res.stdout.strip()}') on attempt {attempt}")
                        return wid
            except Exception as e:
                print(f"  [debug] PID search error: {e}")

        print(f"[*] Window not found yet, retrying ({attempt}/{retries})...")
        time.sleep(delay)

    # Last resort: dump all visible windows so you can pick the right title fragment
    print("\n[!] Could not find window automatically. Here are all visible windows:")
    list_all_windows()
    return None


def focus_window(window_id: str) -> None:
    """Bring the window to the foreground."""
    subprocess.run(["xdotool", "windowfocus", "--sync", window_id], check=True)
    subprocess.run(["xdotool", "windowactivate", "--sync", window_id], check=True)


def send_key(window_id: str, combo: str) -> None:
    """Send a single key or key combination to the specified window."""
    print(f"  → Sending: {combo}")
    subprocess.run(
        ["xdotool", "key", "--window", window_id, "--clearmodifiers", combo],
        check=True
    )


def send_all_keys(window_id: str, combos: list[str], interval: float) -> None:
    """Send all key combinations with a pause between each."""
    for combo in combos:
        focus_window(window_id)
        send_key(window_id, combo)
        time.sleep(interval)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main() -> None:
    # 1. Validate Wine prefix exists
    if not os.path.isdir(WINE_PREFIX):
        print(f"[!] WINEPREFIX not found: {WINE_PREFIX}")
        sys.exit(1)

    # 2. Launch the app
    proc = launch_wine_app()

    # 3. Wait for the app window to appear
    print(f"[*] Waiting {STARTUP_WAIT}s for the application to load...")
    time.sleep(STARTUP_WAIT)

    # 4. Find the window — MetaEditor's title is usually "MetaEditor 5"
    window_title_fragment = "MetaEditor"   # adjust if your window title differs
    window_id = get_window_id(window_title_fragment, proc=proc)

    if not window_id:
        print("[!] Could not find the application window. Aborting.")
        proc.terminate()
        sys.exit(1)

    # 5. Send key combinations
    print(f"\n[*] Sending {len(KEY_COMBOS)} key combination(s) with {INTERVAL}s interval...\n")
    send_all_keys(window_id, KEY_COMBOS, INTERVAL)

    print("\n[*] All keys sent successfully.")

    # 6. Optionally wait for the process to finish
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("[*] Process still running after timeout — leaving it open.")


if __name__ == "__main__":
    main()