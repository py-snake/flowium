"""
Stream Capture Service
Captures frames from YouTube live stream using yt-dlp and ffmpeg.
Includes atomic writes and thread cleanup.
"""
import os
import time
import subprocess
import threading
from pathlib import Path

# Configuration from environment variables
YOUTUBE_URL = os.getenv('YOUTUBE_URL', '')
FRAME_RATE = int(os.getenv('FRAME_RATE', '5'))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', '/shared/frames'))
PROXY_URL = os.getenv('PROXY_URL', '')
COOKIES_FILE = os.getenv('COOKIES_FILE', '')

def log_stderr(process):
    """Read and log stderr in background to prevent pipe blocking"""
    try:
        # iterate lines until the pipe closes
        for line in process.stderr:
            line = line.strip()
            if line and not line.startswith('frame='):  # Skip verbose frame info
                print(f"ffmpeg: {line}")
    except ValueError:
        # Occurs if operation is performed on closed file
        pass
    except Exception as e:
        print(f"Error logging stderr: {e}")

def atomic_file_watcher(temp_path, final_path, stop_event):
    """
    Watch temp file and atomically rename to final path when stable.

    NOTE: We do NOT use shutil.copy here. Copying is slow and non-atomic.
    If ffmpeg writes to the file while we are copying, we get a corrupted image.
    Instead, we wait for stability, then RENAME the file.
    FFmpeg (with -update 1) closes the file handle after every frame,
    so renaming the file out from under it is safe on Linux/Unix;
    FFmpeg will simply create a new file for the next frame.
    """
    last_mtime = 0
    last_stable_time = 0
    # Stability threshold must be less than frame interval (1/5fps = 0.2s)
    # but long enough to ensure write is done.
    stability_threshold = 0.10

    print(f"📁 Starting atomic file watcher: {temp_path.name} -> {final_path.name}")

    while not stop_event.is_set():
        try:
            if temp_path.exists():
                try:
                    stats = temp_path.stat()
                    current_mtime = stats.st_mtime
                    current_size = stats.st_size
                    current_time = time.time()

                    if current_size == 0:
                        # Ignore empty files created during ffmpeg init
                        time.sleep(0.05)
                        continue

                    if current_mtime != last_mtime:
                        # File was modified, reset stability timer
                        last_mtime = current_mtime
                        last_stable_time = current_time
                    elif current_time - last_stable_time >= stability_threshold:
                        # File is stable. Atomically rename it directly.
                        # This moves the inode; it is instant and atomic.
                        try:
                            temp_path.rename(final_path)
                            # Reset times so we don't try to rename non-existent file immediately
                            last_mtime = 0
                        except FileNotFoundError:
                            # FFMPEG might have just started writing a new one, or we just moved it.
                            pass
                        except OSError as e:
                            print(f"⚠️ Rename failed: {e}")

                except FileNotFoundError:
                    # File disappeared (we likely just renamed it), wait for next frame
                    pass

            time.sleep(0.05) # Check every 50ms
        except Exception as e:
            print(f"Watcher error: {e}")
            time.sleep(1)

    print(f"📁 Atomic file watcher stopped")

def get_fresh_stream_url():
    """Get a fresh stream URL from yt-dlp"""
    try:
        # Build yt-dlp command with optional proxy and cookies
        cmd = ['yt-dlp', '-f', 'best', '-g']
        if PROXY_URL:
            cmd.extend(['--proxy', PROXY_URL])
        if COOKIES_FILE and Path(COOKIES_FILE).exists():
            cmd.extend(['--cookies', COOKIES_FILE])
            print(f"🍪 Using cookies from: {COOKIES_FILE}")
        cmd.append(YOUTUBE_URL)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        stream_url = result.stdout.strip()
        print(f"✓ Got fresh stream URL from yt-dlp (length: {len(stream_url)} chars)")
        return stream_url
    except subprocess.TimeoutExpired:
        print("ERROR: yt-dlp timed out after 30 seconds")
        return None
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to get stream URL: {e}")
        if hasattr(e, 'stderr'):
            print(f"stderr: {e.stderr}")
        return None

def capture_stream():
    """Capture frames from YouTube stream using ffmpeg with auto-restart"""
    if not YOUTUBE_URL:
        print("ERROR: YOUTUBE_URL environment variable not set")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting stream capture from: {YOUTUBE_URL}")
    if PROXY_URL:
        print(f"Using proxy: {PROXY_URL}")
    if COOKIES_FILE:
        if Path(COOKIES_FILE).exists():
            print(f"Using cookies file: {COOKIES_FILE}")
        else:
            print(f"⚠️ Cookies file specified but not found: {COOKIES_FILE}")

    # Use temp file for writing, watcher will atomically rename to latest.jpg
    temp_path = OUTPUT_DIR / 'latest_writing.jpg'
    latest_path = OUTPUT_DIR / 'latest.jpg'

    # Configuration
    STALE_THRESHOLD = 30  # Restart if frame is older than 30 seconds
    MAX_RUN_TIME = 1800   # Restart every 30 minutes to refresh URL

    # Start atomic file watcher ONCE per function call
    stop_watcher = threading.Event()
    watcher_thread = threading.Thread(
        target=atomic_file_watcher,
        args=(temp_path, latest_path, stop_watcher),
        daemon=True
    )
    watcher_thread.start()

    try:
        while True:
            # Get fresh stream URL
            stream_url = get_fresh_stream_url()
            if not stream_url:
                print("Retrying in 10 seconds...")
                time.sleep(10)
                continue

            ffmpeg_cmd = [
                'ffmpeg',
                '-re',                  # Read input at native frame rate
                '-i', stream_url,
                '-q:v', '2',            # High quality JPEG
                '-vf', f'fps={FRAME_RATE}',
                '-f', 'image2',
                '-update', '1',
                '-y',
                str(temp_path)
            ]

            if PROXY_URL:
                ffmpeg_cmd.insert(1, '-http_proxy')
                ffmpeg_cmd.insert(2, PROXY_URL)

            print(f"🎬 Starting ffmpeg capture...")

            process = None
            try:
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Start background thread to read stderr
                # We do not join this thread; it dies when process pipe closes
                stderr_thread = threading.Thread(target=log_stderr, args=(process,), daemon=True)
                stderr_thread.start()

                start_time = time.time()
                last_check = start_time
                last_good_frame = None

                while True:
                    # Check if process died
                    if process.poll() is not None:
                        print(f"❌ ffmpeg died with exit code {process.returncode}")
                        break

                    current_time = time.time()

                    # Check health every 5 seconds
                    if current_time - last_check >= 5:
                        if latest_path.exists():
                            file_age = current_time - latest_path.stat().st_mtime

                            if file_age <= 5:
                                # Fresh frames
                                if last_good_frame is None or current_time - last_good_frame > 30:
                                    print(f"✓ Stream healthy (frame age: {file_age:.1f}s)")
                                last_good_frame = current_time
                            elif file_age > STALE_THRESHOLD:
                                print(f"⚠️ Frame is STALE ({file_age:.1f}s old) - restarting ffmpeg...")
                                break # Break inner loop to restart ffmpeg
                        else:
                            print("⏳ Waiting for first frame...")

                        last_check = current_time

                    # Automatic restart
                    if current_time - start_time > MAX_RUN_TIME:
                        print(f"🔄 Auto-restart after {MAX_RUN_TIME}s...")
                        break

                    time.sleep(1)

            finally:
                # Ensure process is killed on break or exception
                if process and process.poll() is None:
                    print("Terminating ffmpeg process...")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

            # Wait a bit before inner loop restart (getting new URL)
            print("⏳ Waiting 3 seconds before restart...")
            time.sleep(3)

    finally:
        # CRITICAL: Stop the watcher thread if this function exits (e.g. via KeyboardInterrupt or crash)
        # Prevents thread leaks if the main block restarts this function.
        print("Stopping file watcher...")
        stop_watcher.set()
        watcher_thread.join(timeout=2)

if __name__ == '__main__':
    while True:
        try:
            capture_stream()
        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            print("Restarting service in 10 seconds...")
            time.sleep(10)
