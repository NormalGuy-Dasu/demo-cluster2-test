import logging
import cv2
import numpy as np
import re

from .base_engine import BaseOcrEngine
from module_2_core_ocr.models import OcrResult, OcrWord, BoundingBox
from .paddle_vietocr import PaddleVietOcrEngine 

logger = logging.getLogger(__name__)

def convert_html_table_to_md(html_str: str) -> str:
    """Dịch mã HTML thô thành Bảng Markdown chuẩn"""
    if not html_str: return ""
    rows = re.findall(r'<tr.*?>(.*?)</tr>', html_str, re.IGNORECASE | re.DOTALL)
    if not rows: return ""
    
    md_table = []
    for i, row in enumerate(rows):
        cells = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', row, re.IGNORECASE | re.DOTALL)
        clean_cells = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]
        md_row = "| " + " | ".join(clean_cells) + " |"
        md_table.append(md_row)
        if i == 0:
            md_table.append("|" + "|".join(["---"] * len(cells)) + "|")
    return "\n" + "\n".join(md_table) + "\n"

def build_markdown_from_words(words_list, img_w, img_h) -> str:
    y_tolerance = max(10, int(img_h * 0.012)) 
    x_gap_threshold = max(30, int(img_w * 0.03)) 
    sorted_words = sorted(words_list, key=lambda w: w.bbox.points[0][1])
    
    md_output = []
    current_line = []
    current_y = None
    
    def flush_line():
        if not current_line: return
        current_line.sort(key=lambda x: x.bbox.points[0][0])
        line_str = ""
        prev_right = None
        for w in current_line:
            txt = w.text.strip()
            if not txt: continue
            left, right = w.bbox.points[0][0], w.bbox.points[1][0]
            if prev_right is not None:
                gap = left - prev_right
                if gap > x_gap_threshold:
                    line_str += " \t|  " + txt 
                else:
                    line_str += " " + txt
            else:
                line_str += txt
            prev_right = right
        if line_str:
            md_output.append(line_str)
        current_line.clear()

    for w in sorted_words:
        if w.block_type == 'table':
            flush_line() 
            html_content = w.metadata.get("html", "")
            md_table = convert_html_table_to_md(html_content)
            md_output.append(md_table)
        else:
            y = w.bbox.points[0][1]
            if current_y is None:
                current_y = y
                current_line.append(w)
            elif abs(y - current_y) <= y_tolerance:
                current_line.append(w)
            else:
                flush_line()
                current_y = y
                current_line.append(w)
                
    flush_line() 
    return "\n".join(md_output)

class DeepdocEngine(BaseOcrEngine):
    def __init__(self, config):
        super().__init__(config)
        
        logger.info("Đang nạp Động cơ Đọc chữ (Paddle + VietOCR)...")
        self.paddle_engine = PaddleVietOcrEngine(config)
        
        # 🌟 KHỞI TẠO LƯỜI BIẾNG (LAZY INITIALIZATION)
        # Mô hình PP-Structure sẽ ngủ đông ở đây, không tốn 1MB VRAM nào.
        self.table_engine = None 

    def _init_paddle_table_engine(self):
        """Hàm đánh thức PP-Structure khi có biến"""
        if self.table_engine is None:
            logger.info("Phát hiện Bảng! Đang đánh thức mô hình PP-Structure (TableSystem)...")
            from paddleocr import PPStructure
            # Chỉ nạp mô hình Table, tắt Layout để tiết kiệm bộ nhớ
            self.table_engine = PPStructure(layout=False, show_log=False)
            
    def _find_tables_opencv(self, image: np.ndarray):
        """Mắt thần OpenCV: Tìm lưới kẻ bảng (Grid Catcher)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -2)
        
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
        
        table_mask = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0.0)
        table_mask = cv2.threshold(table_mask, 50, 255, cv2.THRESH_BINARY)[1]
        
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        table_bboxes = []
        h_img, w_img = image.shape[:2]
        min_area = (h_img * w_img) * 0.015 
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > min_area:
                pad = 10
                x1, y1 = max(0, x-pad), max(0, y-pad)
                x2, y2 = min(w_img, x+w+pad), min(h_img, y+h+pad)
                table_bboxes.append([x1, y1, x2, y2])
                
        return table_bboxes

    def _is_inside_table(self, word_bbox: BoundingBox, table_bbox: BoundingBox) -> bool:
        pts = word_bbox.points
        centroid_x = sum(p[0] for p in pts) / 4.0
        centroid_y = sum(p[1] for p in pts) / 4.0
        t_pts = table_bbox.points
        t_left = min(p[0] for p in t_pts)
        t_right = max(p[0] for p in t_pts)
        t_top = min(p[1] for p in t_pts)
        t_bottom = max(p[1] for p in t_pts)
        return (t_left <= centroid_x <= t_right) and (t_top <= centroid_y <= t_bottom)

    def process_image(self, image: np.ndarray) -> OcrResult:
        logger.info("OpenCV đang quét lưới Bảng biểu (Grid Catcher)...")
        final_words = []
        table_bboxes_objs = []
        
        try:
            # =================================================================
            # GIAI ĐOẠN 1: OPENCV BẮT BẢNG VÀ PADDLE TABLE DỊCH HTML
            # =================================================================
            opencv_boxes = self._find_tables_opencv(image)
            
            if len(opencv_boxes) > 0:
                # Chỉ đánh thức TableSystem nếu OpenCV tìm thấy Bảng
                self._init_paddle_table_engine()
                
            for box in opencv_boxes:
                x1, y1, x2, y2 = box
                bbox_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] 
                table_box = BoundingBox(points=bbox_points)
                table_bboxes_objs.append(table_box)
                
                table_img = image[y1:y2, x1:x2]
                
                # Ném ảnh cho Paddle PP-Structure dịch HTML
                pp_res = self.table_engine(table_img)
                html_content = ""
                
                # PP-Structure trả về list dictionary, lọc tìm key 'html'
                for region in pp_res:
                    if region.get('type') == 'table' and 'res' in region:
                        html_content = region['res'].get('html', '')
                        break
                
                table_word = OcrWord(
                    text="[BẢNG DỮ LIỆU]",
                    confidence=1.0,
                    bbox=table_box,
                    block_type="table",
                    metadata={"html": html_content} 
                )
                final_words.append(table_word)

            # =================================================================
            # GIAI ĐOẠN 2: ĐỘNG CƠ PADDLE QUÉT CHỮ VÙNG AN TOÀN
            # =================================================================
            paddle_result = self.paddle_engine.process_image(image)
            
            for word in paddle_result.words:
                is_inside = any(self._is_inside_table(word.bbox, t_box) for t_box in table_bboxes_objs)
                if not is_inside:
                    final_words.append(word)

            # =================================================================
            # GIAI ĐOẠN 3: ĐÓNG GÓI MARKDOWN
            # =================================================================
            final_words.sort(key=lambda w: w.bbox.points[0][1])
            full_text = "\n".join([w.text for w in final_words if w.text])

            markdown_text = ""
            try:
                h, w = image.shape[:2]
                markdown_text = build_markdown_from_words(final_words, w, h)
            except Exception as e:
                logger.error(f"[LỖI TẠO MARKDOWN]: {e}")

            return OcrResult(
                is_success=True,
                words=final_words,
                full_text=full_text,
                markdown_text=markdown_text 
            )

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            logger.error(f"Lỗi phân tích: \n{err_msg}")
            return OcrResult(is_success=False, words=[], full_text="", error_message=str(e))