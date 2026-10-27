"""Test build of the cluster 2 demo: Module 2 (OCR) + Module 3 (records) on a scanned PDF.

Thin wrapper only: it calls the project code unchanged. One shared OCR engine, one visitor at a time,
a page limit, and every temporary file is deleted after the run (D-019).
"""
import dataclasses
import gc
import json
import resource
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module_3_dynamic_ner"))  # same as PYTHONPATH=module_3_dynamic_ner

HARD_MAX_PAGES = 8
MAX_UPLOAD_MB = 30
ZOOM = 2.0


def rss_mb():
    import psutil

    return round(psutil.Process().memory_info().rss / 1024 / 1024)


def peak_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def seed_models():
    """Put the bundled models where the libraries look for them, so nothing is downloaded at start."""
    src = ROOT / "models" / "paddleocr" / "whl"
    if src.exists():
        shutil.copytree(src, Path.home() / ".paddleocr" / "whl", dirs_exist_ok=True)
    pth = ROOT / "models" / "vgg_seq2seq.pth"
    target = Path(tempfile.gettempdir()) / "vgg_seq2seq.pth"
    if pth.exists() and not target.exists():
        shutil.copy2(pth, target)


@st.cache_resource(show_spinner="Loading the OCR models (first start only)...")
def load_engine():
    t = time.perf_counter()
    seed_models()
    from module_2_core_ocr.config import OcrConfig
    from module_2_core_ocr.ocr_pipeline import OcrPipeline

    cfg = OcrConfig.from_yaml(ROOT / "configs" / "module2_defaults.yaml")
    pipe = OcrPipeline(cfg.active_engine, cfg)
    catalog = json.loads((ROOT / "module_3_dynamic_ner" / "configs" / "document_catalog.json").read_text(encoding="utf-8"))
    return {"pipe": pipe, "catalog": catalog, "lock": threading.Lock(), "load_s": round(time.perf_counter() - t, 1)}


def make_synthetic_pdf(n_pages, land_like=True):
    """A drawn landscape A4 PDF with 28 text lines per page. No personal data."""
    import fitz

    doc = fitz.open()
    for p in range(n_pages):
        page = doc.new_page(width=842, height=595)
        for i in range(28):
            if land_like:
                text = f"Page {p + 1} line {i + 1}: Giay chung nhan quyen su dung dat so {1000 + i * 7} ngay 12/05/2015"
            else:
                text = f"Page {p + 1} line {i + 1}: the weather report for the coast is calm with light winds and clear skies"
            page.insert_text((30, 30 + i * 19), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def run_pipeline(data, max_pages, eng, progress):
    """PDF bytes -> page images -> Module 2 -> Module 3. Returns a dict; never keeps files."""
    import fitz
    from module_3_dynamic_ner.orchestrator.session_manager import SessionManager

    t_all = time.perf_counter()
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:  # noqa: BLE001
        return {"error": "bad_pdf"}
    n = doc.page_count
    if n > max_pages:
        doc.close()
        return {"error": "too_many_pages", "pages": n}

    stream, boxes, t_render, t_ocr = [], 0, 0.0, 0.0
    with tempfile.TemporaryDirectory(prefix="demo_") as work:
        work = Path(work)
        for i in range(n):
            pdir = work / f"p{i + 1}"
            pdir.mkdir()
            t = time.perf_counter()
            doc.load_page(i).get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM)).save(str(pdir / "page.png"))
            t_render += time.perf_counter() - t
            t = time.perf_counter()
            res = eng["pipe"].process_folder(pdir)
            t_ocr += time.perf_counter() - t
            r = res["page.png"]["result"]
            r.page_number = i + 1
            d = json.loads(json.dumps(dataclasses.asdict(r), ensure_ascii=False))  # same shape as the page JSON files
            stream.append({"page_num": i + 1, "full_text": d.get("full_text", ""), "words": d.get("words", [])})
            boxes += len(d.get("words", []))
            del res, r
            shutil.rmtree(pdir)
            progress.progress((i + 1) / n, text=f"Page {i + 1} of {n} done")
        doc.close()
        t = time.perf_counter()
        records = SessionManager(eng["catalog"]).process_document_stream(stream)
        t_m3 = time.perf_counter() - t
        work_left_before_exit = len(list(work.iterdir()))
    total = time.perf_counter() - t_all
    gc.collect()
    return {
        "records": records,
        "stats": {
            "pages": n, "ocr_boxes": boxes, "render_s": round(t_render, 1), "ocr_s": round(t_ocr, 1),
            "module3_s": round(t_m3, 2), "total_s": round(total, 1), "per_page_s": round(total / n, 1),
            "rss_now_mb": rss_mb(), "peak_rss_mb": peak_mb(), "page_files_left": work_left_before_exit,
            "work_dir_deleted": not work.exists(),
        },
    }


def show_result(result, max_pages):
    if result.get("error") == "bad_pdf":
        st.error("Không đọc được file này. Hãy chọn một file PDF hợp lệ.")
        return
    if result.get("error") == "too_many_pages":
        st.error(f"PDF có {result['pages']} trang, vượt giới hạn {max_pages} trang. Hãy cắt bớt rồi thử lại.")
        return
    records = result["records"]
    if not records:
        st.warning("Không nhận ra loại tài liệu nào trong PDF này. Hệ thống chỉ nhận dạng hồ sơ đất đai thuộc các loại đã hỗ trợ, nên không có bản ghi nào để hiển thị.")
    else:
        import pandas as pd

        rows = [{k: (v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v, ensure_ascii=False)) for k, v in rec.items()} for rec in records]
        st.success(f"Tìm thấy {len(records)} bản ghi.")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with st.expander("Diagnostics"):
        st.json(result["stats"])


def guarded_run(data, max_pages, eng):
    """One visitor at a time; the lock is always released."""
    if not eng["lock"].acquire(blocking=False):
        st.warning("Đang xử lý cho một người dùng khác. Vui lòng thử lại sau ít phút.")
        return None
    try:
        bar = st.progress(0.0, text="Starting...")
        result = run_pipeline(data, max_pages, eng, bar)
        bar.empty()
        return result
    finally:
        eng["lock"].release()


st.set_page_config(page_title="Land-document OCR test", layout="wide")
st.title("Land-document OCR (test build)")
st.caption("Modules 2 and 3 on a scanned PDF. Test build: do not upload documents with personal data. Files are processed in a temporary folder that is deleted afterwards.")

engine = load_engine()
max_pages = st.slider("Page limit (test setting)", 1, HARD_MAX_PAGES, 5)
st.caption(f"Models loaded in {engine['load_s']} s. Memory now {rss_mb()} MB, peak {peak_mb()} MB (limit about 3000 MB).")

tab_up, tab_syn = st.tabs(["Upload a PDF", "Synthetic test PDF"])

with tab_up:
    up = st.file_uploader("Scanned land-document PDF", type=["pdf"])
    if up is not None and st.button("Run OCR on the uploaded PDF"):
        if up.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File lớn hơn {MAX_UPLOAD_MB} MB.")
        else:
            res = guarded_run(up.getvalue(), max_pages, engine)
            if res is not None:
                show_result(res, max_pages)

with tab_syn:
    st.write("Builds a drawn PDF with no personal data and runs the same path. Land-like text may give a record; unrelated text gives 0 records and shows the unrecognized-document message.")
    c1, c2, c3 = st.columns(3)
    n_pages = c1.number_input("Pages", 1, HARD_MAX_PAGES, 3)
    repeats = c2.number_input("Repeat runs", 1, 10, 1)
    kind = c3.radio("Text", ["Land-like text", "Unrelated text"])
    if st.button("Run synthetic test"):
        data = make_synthetic_pdf(int(n_pages), kind == "Land-like text")
        history = []
        last = None
        for k in range(int(repeats)):
            last = guarded_run(data, max_pages, engine)
            if last is None:
                break
            if "stats" in last:
                history.append({"run": k + 1, **{x: last["stats"][x] for x in ("pages", "total_s", "per_page_s", "rss_now_mb", "peak_rss_mb", "work_dir_deleted")}})
        if history:
            st.table(history)
        if last is not None:
            show_result(last, max_pages)
