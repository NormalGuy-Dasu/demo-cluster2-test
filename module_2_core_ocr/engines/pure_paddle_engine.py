import numpy as np
from paddleocr import PaddleOCR

from .base_engine import BaseOcrEngine
from module_2_core_ocr.models import OcrResult, OcrWord, BoundingBox
from shared_utils.logger import get_logger

logger = get_logger(__name__)

class PurePaddleEngine(BaseOcrEngine):
    def __init__(self, config):
        self.config = config
        logger.info("[ENGINE] Đang khởi tạo Thuần PaddleOCR (Detection + Recognition siêu nhẹ)...")
        
        # Khởi tạo một Engine duy nhất làm cả 2 nhiệm vụ
        self.ocr_engine = PaddleOCR(
            use_angle_cls=False, # Đã tắt vì Module 1/Utils tự xoay rồi
            lang=self.config.lang,
            show_log=False,
            rec=True, # BẬT nhận diện chữ của Paddle
            det_limit_side_len=self.config.paddle.det_limit_side_len,
            det_db_thresh=self.config.paddle.det_db_thresh,
            det_db_box_thresh=self.config.paddle.det_db_box_thresh,
            use_gpu=self.config.use_gpu
        )
        self.classifier = self.ocr_engine # Để tương thích với hàm xoay ảnh nếu cần

    def process_image(self, image: np.ndarray) -> OcrResult:
        """Thực thi luồng Thuần PaddleOCR: Tốc độ tối đa"""
        try:
            # Chạy thẳng hàm ocr vạn năng của Paddle (nó tự detect, tự cắt, tự recognize)
            results = self.ocr_engine.ocr(image, cls=False)
            
            if not results or not results[0]:
                return OcrResult(is_success=True, words=[], full_text="")

            words = []
            full_text_parts = []
            
            # Bóc tách cấu trúc dữ liệu của PaddleOCR
            for line in results[0]:
                box_coords = line[0]  # Tọa độ 4 góc
                text_tuple = line[1]  # (Chữ, Độ tự tin)
                text = text_tuple[0]
                prob = text_tuple[1]
                
                if not text.strip() or prob is None or np.isnan(prob):
                    continue
                    
                words.append(OcrWord(
                    text=text,
                    confidence=float(prob),
                    bbox=BoundingBox(points=[(int(p[0]), int(p[1])) for p in box_coords])
                ))
                full_text_parts.append(text)
                
            return OcrResult(
                is_success=True,
                words=words,
                full_text="\n".join(full_text_parts)
            )
            
        except Exception as e:
            logger.error(f"Lỗi hệ thống Plugin Thuần PaddleOCR: {e}")
            return OcrResult(is_success=False, words=[], full_text="", error_message=str(e))