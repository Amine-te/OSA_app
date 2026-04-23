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
        """Retrieves exactly the next frame, or skips frames to meet the configured interval."""
        if not self.capture or not self.capture.isOpened():
            return None

        ret = False
        frame = None
        
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                break
                
            self.current_frame_idx += 1
            
            if not enforce_interval or (self.current_frame_idx % self.frame_interval == 0):
                return frame
        return None

    def release(self):
        """Releases the underlying OpenCV resources."""
        if self.capture:
            self.capture.release()
            self.capture = None
