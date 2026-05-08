import cv2
import time

class StreamManager:
    """Handles continuous frame extraction from video files or RTSP feeds."""
    def __init__(self, media_path: str, is_rtsp: bool = False):
        self.media_path = media_path
        self.is_rtsp = is_rtsp
        self.capture = None
        self.frame_interval = 30  # Process every 30th frame by default
        self.current_frame_idx = 0

    def start(self):
        """Initializes the video/RTSP capture object."""
        self.capture = cv2.VideoCapture(self.media_path)
        if not self.capture.isOpened():
            raise ValueError(f"Unable to open media source: {self.media_path}")

    def get_next_frame(self, enforce_interval=True):
        """Retrieves exactly the next frame, or skips frames to meet the configured interval. Auto-reconnects for RTSP."""
        if not self.capture:
            return None

        retries = 0
        max_retries = 5 if self.is_rtsp else 0
        
        while True:
            if not self.capture or not self.capture.isOpened():
                if retries < max_retries:
                    try:
                        self.start()
                    except Exception:
                        pass
                else:
                    return None

            ret, frame = self.capture.read()
            if not ret:
                if self.is_rtsp and retries < max_retries:
                    retries += 1
                    print(f"RTSP Stream lost. Reconnecting... (Attempt {retries}/{max_retries})")
                    self.release()
                    time.sleep(1.0)
                    continue
                else:
                    break
                
            self.current_frame_idx += 1
            retries = 0 # reset retries on successful read
            
            if not enforce_interval or (self.current_frame_idx % self.frame_interval == 0):
                return frame
        return None

    def release(self):
        """Releases the underlying OpenCV resources."""
        if self.capture:
            self.capture.release()
            self.capture = None
