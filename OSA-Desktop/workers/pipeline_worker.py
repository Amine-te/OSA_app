import torch
import cv2
import time
import numpy as np
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

# Import pipeline
from src.pipelines.enhanced_pipeline import EnhancedRetailPipeline
from utils.path_utils import resolve_path

class PipelineWorker(QThread):
    # Signals to communicate securely with the main GUI thread
    started_processing = pyqtSignal()
    frame_processed = pyqtSignal(dict)  # Emits pipeline results dict
    error_occurred = pyqtSignal(str)
    finished_processing = pyqtSignal()

    def __init__(self, media_path: str, config: dict, is_rtsp: bool = False):
        super().__init__()
        self.media_path = str(media_path)
        self.config = config
        self.is_rtsp = is_rtsp
        self.is_running = True
        
    def run(self):
        try:
            self.started_processing.emit()
            
            # Detect compute device
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "MPS"
            elif torch.cuda.is_available():
                self._device = "CUDA"
            else:
                self._device = "CPU"
            
            # Use dynamic paths relative to project root
            yolo_product = str(resolve_path(self.config.get("models", {}).get("yolo_product", "")))
            cnn_class = str(resolve_path(self.config.get("models", {}).get("cnn_class", "")))
            yolo_void = str(resolve_path(self.config.get("models", {}).get("yolo_void", "")))
            
            print(f"Loading models from: {yolo_product}")
            
            pipeline = EnhancedRetailPipeline(
                yolo_model_path=yolo_product,
                cnn_model_path=cnn_class,
                void_model_path=yolo_void,
                class_names=self.config.get("class_names", []),
                confidence_threshold=self.config.get("thresholds", {}).get("confidence", 0.5),
                void_confidence_threshold=self.config.get("thresholds", {}).get("void_confidence", 0.5)
            )

            # Determine if media is an image or video/RTSP
            path_lower = self.media_path.lower()
            if not self.is_rtsp and (path_lower.endswith('.jpg') or path_lower.endswith('.jpeg') or path_lower.endswith('.png')):
                self._process_image(pipeline)
            else:
                self._process_video_stream(pipeline)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))
        finally:
            self.finished_processing.emit()

    def _process_image(self, pipeline):
        t0 = time.time()
        results = pipeline.detect_and_classify_complete(self.media_path)
        results['inference_time_ms'] = (time.time() - t0) * 1000
        results['device'] = self._device
        results = self._draw_annotations(results)
        self.frame_processed.emit(results)

    def _process_video_stream(self, pipeline):
        from workers.stream_manager import StreamManager
        
        stream_manager = StreamManager(self.media_path, is_rtsp=self.is_rtsp)
        stream_manager.start()
        
        while self.is_running:
            frame = stream_manager.get_next_frame(enforce_interval=True)
            if frame is None:
                break
                
            t0 = time.time()
            results = pipeline.detect_and_classify_complete(frame)
            results['inference_time_ms'] = (time.time() - t0) * 1000
            results['device'] = self._device
            results = self._draw_annotations(results)
            self.frame_processed.emit(results)
            
        stream_manager.release()

    def stop(self):
        self.is_running = False
        self.wait()

    def _draw_annotations(self, results):
        results['raw_image'] = results['image'].copy()
        image = results['image'].copy()
        
        color_map = {
            "coca_cola": (255, 0, 0),    # Red
            "pepsi": (0, 0, 255),        # Blue
            "fanta": (255, 165, 0),      # Orange
            "sprite": (0, 255, 0),       # Green
        }
        
        for det in results.get('product_detections', []):
            x1, y1, x2, y2 = det['bbox']
            subclass = det['subclass']
            conf = det['combined_confidence']
            
            color = color_map.get(subclass, (0, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f'{subclass} {conf:.2f}', (x1, max(0, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
        for i, void in enumerate(results.get('void_detections', [])):
            x1, y1, x2, y2 = void['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2, lineType=cv2.LINE_4)
            
            if i < len(results.get('void_analysis', [])):
                void_info = results['void_analysis'][i]
                if void_info.get('final_assignment'):
                    assign = void_info['final_assignment']
                    ptype = assign.get('product_type', 'unknown')
                    cv2.putText(image, f"VOID: {ptype}", (x1, max(0, y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                
        results['image'] = image
        return results
