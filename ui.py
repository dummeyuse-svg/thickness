add above def main(): 

  also make changes for ui and cli both usage 

                   elif choice == "5":
    break

elif choice == "6":
    PCB_UI()

else:
    print("  Invalid choice. Enter 1–6.")

# ──────────────────────────────────────────────
# UI MODE (OpenCV + Tkinter Hybrid)
# ──────────────────────────────────────────────

import threading
import tkinter as tk

class PCB_UI:
    def __init__(self):
        self.cam = None
        self.baseline = None
        self.current_rgb = None
        self.current_gray = None

        self.running = True

        self.init_backend()
        self.init_ui()

    # ─────────────────────────────
    # BACKEND INIT
    # ─────────────────────────────
    def init_backend(self):
        load_roi_config()

        if CAMERA_AVAILABLE:
            self.cam = init_camera()

        try:
            self.baseline = load_baseline()
        except:
            print("[WARN] No baseline found. Please calibrate first.")

    # ─────────────────────────────
    # UI WINDOW (Tkinter panel)
    # ─────────────────────────────
    def init_ui(self):
        self.root = tk.Tk()
        self.root.title("PCB Inspection Control Panel")

        # ROI Inputs
        tk.Label(self.root, text="ROI X1").grid(row=0, column=0)
        tk.Label(self.root, text="ROI Y1").grid(row=1, column=0)
        tk.Label(self.root, text="ROI X2").grid(row=2, column=0)
        tk.Label(self.root, text="ROI Y2").grid(row=3, column=0)

        self.x1 = tk.Entry(self.root)
        self.y1 = tk.Entry(self.root)
        self.x2 = tk.Entry(self.root)
        self.y2 = tk.Entry(self.root)

        self.x1.grid(row=0, column=1)
        self.y1.grid(row=1, column=1)
        self.x2.grid(row=2, column=1)
        self.y2.grid(row=3, column=1)

        # Buttons
        tk.Button(self.root, text="Capture", command=self.capture).grid(row=4, column=0)
        tk.Button(self.root, text="Inspect", command=self.inspect).grid(row=4, column=1)
        tk.Button(self.root, text="Draw ROI", command=self.draw_roi).grid(row=5, column=0)
        tk.Button(self.root, text="Apply ROI", command=self.apply_roi).grid(row=5, column=1)

        self.status = tk.Label(self.root, text="Status: Idle")
        self.status.grid(row=6, column=0, columnspan=2)

        threading.Thread(target=self.preview_loop, daemon=True).start()

        self.root.mainloop()

    # ─────────────────────────────
    # LIVE PREVIEW LOOP
    # ─────────────────────────────
    def preview_loop(self):
        while self.running:
            if self.cam:
                rgb, gray = capture_from_camera(self.cam)
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Draw ROI box
                cv2.rectangle(
                    frame,
                    (CONFIG["ROI_X_START"], CONFIG["ROI_Y_START"]),
                    (CONFIG["ROI_X_END"], CONFIG["ROI_Y_END"]),
                    (0, 255, 255),
                    2
                )

                cv2.imshow("LIVE PREVIEW", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    self.capture()
                elif key == ord('i'):
                    self.inspect()
                elif key == ord('r'):
                    self.draw_roi()
                elif key == 27:
                    self.running = False
                    break

        cv2.destroyAllWindows()

    # ─────────────────────────────
    # CAPTURE IMAGE
    # ─────────────────────────────
    def capture(self):
        if not self.cam:
            return

        self.current_rgb, self.current_gray = capture_from_camera(self.cam)

        bgr = cv2.cvtColor(self.current_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("CAPTURED IMAGE", bgr)

        self.status.config(text="Captured image")

    # ─────────────────────────────
    # INSPECT
    # ─────────────────────────────
    def inspect(self):
        if self.current_gray is None or self.baseline is None:
            self.status.config(text="Error: No image or baseline")
            return

        passed = inspect_image(
            self.current_rgb,
            self.current_gray,
            self.baseline,
            source_label="UI"
        )

        self.status.config(text=f"Result: {'PASS' if passed else 'FAIL'}")

    # ─────────────────────────────
    # DRAW ROI (interactive)
    # ─────────────────────────────
    def draw_roi(self):
        if self.current_rgb is None:
            self.status.config(text="Capture image first")
            return

        result = select_roi_interactively(self.current_rgb)
        if result:
            x1, y1, x2, y2 = result

            CONFIG["ROI_X_START"] = x1
            CONFIG["ROI_Y_START"] = y1
            CONFIG["ROI_X_END"]   = x2
            CONFIG["ROI_Y_END"]   = y2

            save_roi_config()

            self.status.config(text="ROI Updated")

    # ─────────────────────────────
    # APPLY MANUAL ROI
    # ─────────────────────────────
    def apply_roi(self):
        try:
            CONFIG["ROI_X_START"] = int(self.x1.get())
            CONFIG["ROI_Y_START"] = int(self.y1.get())
            CONFIG["ROI_X_END"]   = int(self.x2.get())
            CONFIG["ROI_Y_END"]   = int(self.y2.get())

            save_roi_config()

            self.status.config(text="Manual ROI Applied")
        except:
            self.status.config(text="Invalid ROI input")


# ─────────────────────────────
# RUN UI INSTEAD OF CLI
# ─────────────────────────────
if __name__ == "__main__":
    PCB_UI()




#!/usr/bin/env python3
"""
PCB Warpage / Uplift Detection System — PyQt5 UI
All detection logic unchanged. UI wrapped on top.
"""

import cv2
import numpy as np
import time
import os
import json
import sys
import threading
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QLineEdit, QFileDialog,
    QStatusBar, QSplitter, QFrame, QScrollArea, QMessageBox,
    QProgressBar, QSpinBox, QDoubleSpinBox, QFormLayout, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

# ──────────────────────────────────────────────
# CONFIGURATION  (unchanged from original)
# ──────────────────────────────────────────────
CONFIG = {
    "CAPTURE_RESOLUTION": (2304, 1296),
    "LENS_POSITION": 2.0,
    "EXPOSURE_TIME": 5000,
    "TARGET_WIDTH": 2000,

    "ROI_Y_START": 600,
    "ROI_Y_END":   900,
    "ROI_X_START": 100,
    "ROI_X_END":   1900,

    "UPLIFT_THRESHOLD_PX": 6,
    "MIN_FAIL_COLUMNS":    10,
    "MAX_MM_THRESHOLD":    0.3,

    "BASELINE_FILE":   "baseline.json",
    "ROI_CONFIG_FILE": "roi_config.json",
    "LOG_DIR":         "inspection_logs",

    "BLUR_KERNEL": 5,
    "CANNY_LOW":   30,
    "CANNY_HIGH":  100,

    "PX_PER_MM": 10.0,
    "CALIBRATION_IMAGES": 5,
}

# ──────────────────────────────────────────────
# DETECTION LOGIC  (unchanged from original)
# ──────────────────────────────────────────────

def normalize_image(bgr_img):
    h, w = bgr_img.shape[:2]
    scale = CONFIG["TARGET_WIDTH"] / w
    return cv2.resize(bgr_img, (CONFIG["TARGET_WIDTH"], int(h * scale)),
                      interpolation=cv2.INTER_AREA)

def load_image_from_file(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    bgr  = normalize_image(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, gray

def load_roi_config():
    if os.path.exists(CONFIG["ROI_CONFIG_FILE"]):
        with open(CONFIG["ROI_CONFIG_FILE"]) as f:
            roi = json.load(f)
        for key in ("ROI_X_START", "ROI_X_END", "ROI_Y_START", "ROI_Y_END"):
            if key in roi:
                CONFIG[key] = roi[key]

def save_roi_config():
    with open(CONFIG["ROI_CONFIG_FILE"], "w") as f:
        json.dump({
            "ROI_X_START": CONFIG["ROI_X_START"],
            "ROI_X_END":   CONFIG["ROI_X_END"],
            "ROI_Y_START": CONFIG["ROI_Y_START"],
            "ROI_Y_END":   CONFIG["ROI_Y_END"],
        }, f, indent=2)

def get_edge_profile(gray_img):
    roi = gray_img[
        CONFIG["ROI_Y_START"]:CONFIG["ROI_Y_END"],
        CONFIG["ROI_X_START"]:CONFIG["ROI_X_END"]
    ]
    blurred = cv2.GaussianBlur(roi, (CONFIG["BLUR_KERNEL"], CONFIG["BLUR_KERNEL"]), 0)
    edges   = cv2.Canny(blurred, CONFIG["CANNY_LOW"], CONFIG["CANNY_HIGH"])

    roi_height = CONFIG["ROI_Y_END"] - CONFIG["ROI_Y_START"]
    profile    = np.full(edges.shape[1], fill_value=float(roi_height), dtype=np.float32)

    for col in range(edges.shape[1]):
        rows = np.where(edges[:, col] > 0)[0]
        if len(rows):
            profile[col] = np.median(rows[:5])

    profile = profile + CONFIG["ROI_Y_START"]
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).flatten()
    return profile

def analyze_uplift(current_profile, baseline_profile):
    if len(current_profile) != len(baseline_profile):
        baseline_profile = np.interp(
            np.linspace(0, 1, len(current_profile)),
            np.linspace(0, 1, len(baseline_profile)),
            baseline_profile
        )

    offset          = np.median(current_profile - baseline_profile)
    current_profile = current_profile - offset

    diff     = baseline_profile - current_profile
    diff_mm  = diff / CONFIG["PX_PER_MM"]
    abs_diff = np.abs(diff)
    abs_diff = cv2.GaussianBlur(abs_diff.reshape(-1, 1), (1, 11), 0).flatten()
    abs_mm   = abs_diff / CONFIG["PX_PER_MM"]

    flagged    = abs_diff > CONFIG["UPLIFT_THRESHOLD_PX"]
    max_abs_mm = float(np.max(abs_mm))
    max_raw_mm = float(diff_mm[np.argmax(abs_diff)])

    fail_regions = []
    in_region    = False
    region_start = 0
    total_cols   = len(flagged)

    for i in range(total_cols + 1):
        is_flagged = (i < total_cols) and flagged[i]
        if is_flagged and not in_region:
            in_region    = True
            region_start = i
        elif not is_flagged and in_region:
            in_region = False
            length    = i - region_start
            if length >= CONFIG["MIN_FAIL_COLUMNS"]:
                seg    = diff[region_start:i]
                seg_mm = diff_mm[region_start:i]
                fail_regions.append({
                    "col_start":     region_start,
                    "col_end":       i,
                    "max_uplift_px": float(np.max(np.abs(seg))),
                    "max_uplift_mm": float(np.max(np.abs(seg_mm))),
                    "direction":     "lifted" if np.mean(seg) > 0 else "sunken",
                    "x_start_mm":    (region_start / total_cols) * 200.0,
                    "x_end_mm":      (i / total_cols) * 200.0,
                })

    passed = not (len(fail_regions) > 0 or max_abs_mm > CONFIG["MAX_MM_THRESHOLD"])
    return passed, fail_regions, diff, diff_mm, abs_diff, max_abs_mm, max_raw_mm

def build_annotated_image(color_img, current_profile, baseline_profile,
                           abs_diff, fail_regions, passed):
    annotated = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    roi_x0    = CONFIG["ROI_X_START"]

    for col_idx, by in enumerate(baseline_profile):
        x, y = col_idx + roi_x0, int(by)
        if 0 <= y < annotated.shape[0] and 0 <= x < annotated.shape[1]:
            annotated[y, x] = (0, 255, 0)

    for col_idx, cy in enumerate(current_profile):
        x, y  = col_idx + roi_x0, int(cy)
        color = (0, 0, 255) if abs_diff[col_idx] > CONFIG["UPLIFT_THRESHOLD_PX"] else (0, 255, 255)
        if 0 <= y < annotated.shape[0] and 0 <= x < annotated.shape[1]:
            annotated[y, x] = color

    cv2.rectangle(annotated,
                  (CONFIG["ROI_X_START"], CONFIG["ROI_Y_START"]),
                  (CONFIG["ROI_X_END"],   CONFIG["ROI_Y_END"]),
                  (0, 165, 255), 2)

    label = "PASS" if passed else f"FAIL — {len(fail_regions)} region(s)"
    cv2.putText(annotated, label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                (0, 200, 0) if passed else (0, 0, 255), 5)

    y_off = 160
    for r in fail_regions:
        direction = "UP" if r["direction"] == "lifted" else "DOWN"
        cv2.putText(annotated,
                    f"  {direction}  {r['max_uplift_mm']:.2f}mm  "
                    f"@ {r['x_start_mm']:.0f}-{r['x_end_mm']:.0f}mm",
                    (50, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        y_off += 55

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

def load_baseline():
    if not os.path.exists(CONFIG["BASELINE_FILE"]):
        raise FileNotFoundError("No baseline.json — calibrate first.")
    with open(CONFIG["BASELINE_FILE"]) as f:
        data = json.load(f)
    return np.array(data["baseline_per_col"]), data

# ──────────────────────────────────────────────
# WORKER SIGNALS
# ──────────────────────────────────────────────
class WorkerSignals(QObject):
    status    = pyqtSignal(str)
    image     = pyqtSignal(np.ndarray)   # annotated result image
    result    = pyqtSignal(bool, list, float)  # passed, fail_regions, max_mm
    progress  = pyqtSignal(int)
    error     = pyqtSignal(str)
    done      = pyqtSignal()

# ──────────────────────────────────────────────
# ROI DRAW WIDGET  (click-drag on image)
# ──────────────────────────────────────────────
class ROIDrawWidget(QLabel):
    roi_selected = pyqtSignal(int, int, int, int)   # x1 y1 x2 y2 in image coords

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self._orig_rgb   = None
        self._scale      = 1.0
        self._offset_x   = 0
        self._offset_y   = 0
        self._start      = None
        self._end        = None
        self._drawing    = False
        self.setStyleSheet("border: 2px solid #555; background: #1a1a1a;")
        self.setText("Load an image to draw ROI")
        self.setStyleSheet("border: 2px solid #555; background: #1a1a1a; color: #888;")

    def set_image(self, rgb_img):
        self._orig_rgb = rgb_img.copy()
        self._start    = None
        self._end      = None
        self._refresh()

    def _img_coords(self, wx, wy):
        """Convert widget coords → original image coords."""
        ix = (wx - self._offset_x) / self._scale
        iy = (wy - self._offset_y) / self._scale
        return int(np.clip(ix, 0, self._orig_rgb.shape[1])), \
               int(np.clip(iy, 0, self._orig_rgb.shape[0]))

    def _refresh(self):
        if self._orig_rgb is None:
            return
        display = self._orig_rgb.copy()

        # Draw existing ROI from CONFIG in orange
        ih, iw = display.shape[:2]
        x1c = CONFIG["ROI_X_START"]
        y1c = CONFIG["ROI_Y_START"]
        x2c = CONFIG["ROI_X_END"]
        y2c = CONFIG["ROI_Y_END"]
        cv2.rectangle(display, (x1c, y1c), (x2c, y2c), (255, 140, 0), 2)
        cv2.putText(display, "Current ROI", (x1c, max(y1c - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 0), 2)

        # Draw new selection in green
        if self._start and self._end:
            ix1 = min(self._start[0], self._end[0])
            iy1 = min(self._start[1], self._end[1])
            ix2 = max(self._start[0], self._end[0])
            iy2 = max(self._start[1], self._end[1])
            cv2.rectangle(display, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
            cv2.putText(display,
                        f"New: ({ix1},{iy1}) → ({ix2},{iy2})  {ix2-ix1}×{iy2-iy1}px",
                        (ix1, max(iy1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self._show(display)

    def _show(self, rgb):
        h, w = rgb.shape[:2]
        ww, wh = self.width(), self.height()
        scale = min(ww / w, wh / h, 1.0)
        self._scale    = scale
        nw, nh         = int(w * scale), int(h * scale)
        self._offset_x = (ww - nw) // 2
        self._offset_y = (wh - nh) // 2
        resized = cv2.resize(rgb, (nw, nh))
        qimg    = QImage(resized.data, nw, nh, nw * 3, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

    def mousePressEvent(self, e):
        if self._orig_rgb is None or e.button() != Qt.LeftButton:
            return
        self._drawing = True
        self._start   = self._img_coords(e.x(), e.y())
        self._end     = self._start

    def mouseMoveEvent(self, e):
        if self._drawing:
            self._end = self._img_coords(e.x(), e.y())
            self._refresh()

    def mouseReleaseEvent(self, e):
        if self._drawing and e.button() == Qt.LeftButton:
            self._drawing = False
            self._end     = self._img_coords(e.x(), e.y())
            self._refresh()

    def confirm_roi(self):
        if self._start and self._end:
            x1 = min(self._start[0], self._end[0])
            y1 = min(self._start[1], self._end[1])
            x2 = max(self._start[0], self._end[0])
            y2 = max(self._start[1], self._end[1])
            if x2 > x1 and y2 > y1:
                self.roi_selected.emit(x1, y1, x2, y2)
                return True
        return False

    def resizeEvent(self, e):
        self._refresh()

# ──────────────────────────────────────────────
# IMAGE DISPLAY LABEL  (fit-to-box, aspect-preserved)
# ──────────────────────────────────────────────
class ImageDisplay(QLabel):
    def __init__(self, placeholder="No image"):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("border: 1px solid #444; background: #111; color: #666;")
        self._pixmap_orig = None

    def set_rgb(self, rgb_img):
        h, w = rgb_img.shape[:2]
        qimg = QImage(rgb_img.data, w, h, w * 3, QImage.Format_RGB888)
        self._pixmap_orig = QPixmap.fromImage(qimg)
        self._fit()

    def _fit(self):
        if self._pixmap_orig:
            self.setPixmap(self._pixmap_orig.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        self._fit()

# ──────────────────────────────────────────────
# CAMERA PREVIEW THREAD
# ──────────────────────────────────────────────
class CameraPreviewThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, cam):
        super().__init__()
        self._cam     = cam
        self._running = True

    def run(self):
        while self._running:
            try:
                frame = self._cam.capture_array()
                # Draw ROI on preview
                preview = frame.copy()
                cv2.rectangle(preview,
                              (CONFIG["ROI_X_START"] // 4, CONFIG["ROI_Y_START"] // 4),
                              (CONFIG["ROI_X_END"]   // 4, CONFIG["ROI_Y_END"]   // 4),
                              (255, 140, 0), 2)
                self.frame_ready.emit(preview)
            except Exception:
                pass
            self.msleep(100)

    def stop(self):
        self._running = False
        self.wait()

# ──────────────────────────────────────────────
# MAIN WINDOW
# ──────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB Warpage Detection System")
        self.resize(1400, 820)
        self._apply_theme()

        self._cam             = None
        self._preview_thread  = None
        self._baseline_profile = None
        self._last_rgb        = None   # last captured/loaded image for ROI drawing
        self._cal_profiles    = []

        load_roi_config()

        self._build_ui()
        self._load_baseline_silent()

    # ── THEME ──────────────────────────────────
    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #1e1e1e; color: #e0e0e0; }
            QGroupBox {
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 12px;
                color: #aaa;
                padding: 6px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #aaa; }
            QPushButton {
                background: #2d2d2d;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 7px 14px;
                color: #ddd;
                font-size: 12px;
            }
            QPushButton:hover  { background: #383838; border-color: #666; }
            QPushButton:pressed { background: #222; }
            QPushButton#btn_capture  { background: #1a3a5c; border-color: #2a6aa0; }
            QPushButton#btn_capture:hover { background: #1f4a75; }
            QPushButton#btn_calibrate { background: #2a4a2a; border-color: #4a8a4a; }
            QPushButton#btn_calibrate:hover { background: #335533; }
            QPushButton#btn_inspect  { background: #4a2a10; border-color: #9a5a20; }
            QPushButton#btn_inspect:hover { background: #5a3518; }
            QPushButton#btn_roi_confirm { background: #3a2a5c; border-color: #6a4aaa; }
            QPushButton#btn_roi_confirm:hover { background: #4a3a72; }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background: #2a2a2a; border: 1px solid #444;
                border-radius: 3px; padding: 4px; color: #ddd;
            }
            QTabWidget::pane { border: 1px solid #3a3a3a; }
            QTabBar::tab {
                background: #2a2a2a; border: 1px solid #3a3a3a;
                padding: 6px 16px; color: #888;
            }
            QTabBar::tab:selected { background: #1e1e1e; color: #ddd; border-bottom: none; }
            QLabel#result_label {
                font-size: 22px; font-weight: bold;
                padding: 12px; border-radius: 6px;
                border: 2px solid #444;
            }
            QProgressBar {
                border: 1px solid #444; border-radius: 3px;
                background: #2a2a2a; text-align: center; color: #ddd;
            }
            QProgressBar::chunk { background: #2a6aa0; }
        """)

    # ── BUILD UI ────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── LEFT: preview + captured image ──────
        left = QVBoxLayout()
        left.setSpacing(6)

        # Live preview
        grp_preview = QGroupBox("Live Camera Preview")
        vp = QVBoxLayout(grp_preview)
        self._preview_display = ImageDisplay("Camera not started")
        self._preview_display.setMinimumSize(480, 270)
        vp.addWidget(self._preview_display)
        btn_start_cam = QPushButton("▶  Start Camera Preview")
        btn_start_cam.clicked.connect(self._start_camera)
        vp.addWidget(btn_start_cam)
        left.addWidget(grp_preview, stretch=1)

        # Captured / result image
        grp_captured = QGroupBox("Captured / Result Image")
        vc = QVBoxLayout(grp_captured)
        self._captured_display = ImageDisplay("No image captured yet")
        self._captured_display.setMinimumSize(480, 270)
        vc.addWidget(self._captured_display)
        left.addWidget(grp_captured, stretch=1)

        root.addLayout(left, stretch=3)

        # ── RIGHT: controls ──────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        tabs = QTabWidget()
        tabs.addTab(self._build_tab_main(),     "  Main  ")
        tabs.addTab(self._build_tab_roi(),      "  ROI  ")
        tabs.addTab(self._build_tab_settings(), "  Settings  ")
        right.addWidget(tabs, stretch=1)

        # Result panel
        grp_result = QGroupBox("Result")
        vr = QVBoxLayout(grp_result)
        self._result_label = QLabel("—")
        self._result_label.setObjectName("result_label")
        self._result_label.setAlignment(Qt.AlignCenter)
        self._result_label.setMinimumHeight(60)
        vr.addWidget(self._result_label)
        self._detail_label = QLabel("")
        self._detail_label.setWordWrap(True)
        self._detail_label.setStyleSheet("color: #aaa; font-size: 11px;")
        vr.addWidget(self._detail_label)
        right.addWidget(grp_result)

        # Status bar
        self._status = QStatusBar()
        self._status.setStyleSheet("color: #888; font-size: 11px;")
        self.setStatusBar(self._status)
        self._set_status("Ready.")

        root.addLayout(right, stretch=2)

    def _build_tab_main(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(10)
        v.setContentsMargins(8, 12, 8, 8)

        # Capture
        grp = QGroupBox("1 · Capture Image")
        vg  = QVBoxLayout(grp)
        btn_cap = QPushButton("📷  Capture from Camera")
        btn_cap.setObjectName("btn_capture")
        btn_cap.clicked.connect(self._capture_camera)
        btn_load = QPushButton("📂  Load from File")
        btn_load.clicked.connect(self._load_file)
        vg.addWidget(btn_cap)
        vg.addWidget(btn_load)
        v.addWidget(grp)

        # Calibrate
        grp2 = QGroupBox("2 · Calibrate  (bare jig — no PCB)")
        vg2  = QVBoxLayout(grp2)
        self._cal_progress = QProgressBar()
        self._cal_progress.setRange(0, CONFIG["CALIBRATION_IMAGES"])
        self._cal_progress.setValue(0)
        self._cal_progress.setFormat(f"0 / {CONFIG['CALIBRATION_IMAGES']} images")
        self._cal_progress.setVisible(False)
        vg2.addWidget(self._cal_progress)
        btn_cal = QPushButton("⚙  Add Calibration Image")
        btn_cal.setObjectName("btn_calibrate")
        btn_cal.setToolTip("Load/capture bare-jig images one at a time. "
                           f"Need {CONFIG['CALIBRATION_IMAGES']} to complete.")
        btn_cal.clicked.connect(self._add_calibration_image)
        btn_cal_finish = QPushButton("✔  Finish Calibration")
        btn_cal_finish.setObjectName("btn_calibrate")
        btn_cal_finish.clicked.connect(self._finish_calibration)
        btn_cal_reset = QPushButton("✖  Reset Calibration")
        btn_cal_reset.clicked.connect(self._reset_calibration)
        self._cal_count_label = QLabel("No calibration images yet.")
        self._cal_count_label.setStyleSheet("color: #888; font-size: 11px;")
        vg2.addWidget(btn_cal)
        vg2.addWidget(btn_cal_finish)
        vg2.addWidget(btn_cal_reset)
        vg2.addWidget(self._cal_progress)
        vg2.addWidget(self._cal_count_label)
        v.addWidget(grp2)

        # Inspect
        grp3 = QGroupBox("3 · Inspect")
        vg3  = QVBoxLayout(grp3)
        btn_insp_cam = QPushButton("🔍  Inspect — Capture Now")
        btn_insp_cam.setObjectName("btn_inspect")
        btn_insp_cam.clicked.connect(self._inspect_camera)
        btn_insp_file = QPushButton("🔍  Inspect — Load File")
        btn_insp_file.setObjectName("btn_inspect")
        btn_insp_file.clicked.connect(self._inspect_file)
        self._baseline_status = QLabel("Baseline: not loaded")
        self._baseline_status.setStyleSheet("color: #888; font-size: 11px;")
        vg3.addWidget(btn_insp_cam)
        vg3.addWidget(btn_insp_file)
        vg3.addWidget(self._baseline_status)
        v.addWidget(grp3)

        v.addStretch()
        return w

    def _build_tab_roi(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(8)
        v.setContentsMargins(8, 12, 8, 8)

        # Manual coordinate entry
        grp_manual = QGroupBox("Manual ROI Coordinates")
        form = QFormLayout(grp_manual)
        self._roi_x1 = QSpinBox(); self._roi_x1.setRange(0, 9999); self._roi_x1.setValue(CONFIG["ROI_X_START"])
        self._roi_y1 = QSpinBox(); self._roi_y1.setRange(0, 9999); self._roi_y1.setValue(CONFIG["ROI_Y_START"])
        self._roi_x2 = QSpinBox(); self._roi_x2.setRange(0, 9999); self._roi_x2.setValue(CONFIG["ROI_X_END"])
        self._roi_y2 = QSpinBox(); self._roi_y2.setRange(0, 9999); self._roi_y2.setValue(CONFIG["ROI_Y_END"])
        form.addRow("X Start:", self._roi_x1)
        form.addRow("Y Start:", self._roi_y1)
        form.addRow("X End:",   self._roi_x2)
        form.addRow("Y End:",   self._roi_y2)
        btn_apply_manual = QPushButton("Apply Manual Coordinates")
        btn_apply_manual.clicked.connect(self._apply_manual_roi)
        form.addRow(btn_apply_manual)
        v.addWidget(grp_manual)

        # Draw on image
        grp_draw = QGroupBox("Draw ROI on Image  (click + drag)")
        vd = QVBoxLayout(grp_draw)

        self._roi_widget = ROIDrawWidget()
        self._roi_widget.setMinimumHeight(280)
        self._roi_widget.roi_selected.connect(self._on_roi_drawn)
        vd.addWidget(self._roi_widget)

        btn_row = QHBoxLayout()
        btn_load_for_roi = QPushButton("Load Image for Drawing")
        btn_load_for_roi.clicked.connect(self._load_for_roi)
        btn_confirm_roi = QPushButton("✔  Confirm Drawn ROI")
        btn_confirm_roi.setObjectName("btn_roi_confirm")
        btn_confirm_roi.clicked.connect(self._confirm_drawn_roi)
        btn_row.addWidget(btn_load_for_roi)
        btn_row.addWidget(btn_confirm_roi)
        vd.addLayout(btn_row)

        self._roi_status = QLabel(self._roi_str())
        self._roi_status.setStyleSheet("color: #888; font-size: 11px;")
        vd.addWidget(self._roi_status)
        v.addWidget(grp_draw)

        v.addStretch()
        return w

    def _build_tab_settings(self):
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(12, 16, 12, 8)
        form.setSpacing(10)

        def spin(val, lo, hi, step=1):
            s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
            return s

        def dspin(val, lo, hi, step=0.05, dec=2):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val)
            s.setSingleStep(step); s.setDecimals(dec)
            return s

        self._s_threshold_px  = spin(CONFIG["UPLIFT_THRESHOLD_PX"], 1, 50)
        self._s_min_cols      = spin(CONFIG["MIN_FAIL_COLUMNS"], 1, 500)
        self._s_max_mm        = dspin(CONFIG["MAX_MM_THRESHOLD"], 0.01, 10.0)
        self._s_px_per_mm     = dspin(CONFIG["PX_PER_MM"], 0.1, 100.0, step=0.1)
        self._s_canny_low     = spin(CONFIG["CANNY_LOW"], 1, 255)
        self._s_canny_high    = spin(CONFIG["CANNY_HIGH"], 1, 255)
        self._s_cal_images    = spin(CONFIG["CALIBRATION_IMAGES"], 1, 20)
        self._s_blur          = spin(CONFIG["BLUR_KERNEL"], 1, 31, step=2)

        form.addRow("Uplift threshold (px):",       self._s_threshold_px)
        form.addRow("Min fail columns:",             self._s_min_cols)
        form.addRow("Max deviation (mm):",           self._s_max_mm)
        form.addRow("px per mm:",                    self._s_px_per_mm)
        form.addRow("Canny low threshold:",          self._s_canny_low)
        form.addRow("Canny high threshold:",         self._s_canny_high)
        form.addRow("Calibration images needed:",    self._s_cal_images)
        form.addRow("Blur kernel size:",             self._s_blur)

        btn_apply = QPushButton("Apply Settings")
        btn_apply.clicked.connect(self._apply_settings)
        form.addRow(btn_apply)

        note = QLabel("Changes apply immediately to next inspection.\n"
                      "Re-calibrate after changing blur/Canny settings.")
        note.setStyleSheet("color: #666; font-size: 10px;")
        note.setWordWrap(True)
        form.addRow(note)

        return w

    # ── HELPERS ────────────────────────────────
    def _set_status(self, msg):
        self._status.showMessage(msg)

    def _roi_str(self):
        return (f"Current ROI — "
                f"X: {CONFIG['ROI_X_START']}–{CONFIG['ROI_X_END']}  "
                f"Y: {CONFIG['ROI_Y_START']}–{CONFIG['ROI_Y_END']}")

    def _update_roi_display(self):
        self._roi_status.setText(self._roi_str())
        self._roi_x1.setValue(CONFIG["ROI_X_START"])
        self._roi_y1.setValue(CONFIG["ROI_Y_START"])
        self._roi_x2.setValue(CONFIG["ROI_X_END"])
        self._roi_y2.setValue(CONFIG["ROI_Y_END"])
        if self._last_rgb is not None:
            self._roi_widget.set_image(self._last_rgb)

    def _show_result(self, passed, fail_regions, max_mm):
        if passed:
            self._result_label.setText("✅  PASS")
            self._result_label.setStyleSheet(
                "font-size:22px; font-weight:bold; padding:12px; border-radius:6px;"
                "border:2px solid #4a8a4a; color:#6fcf6f; background:#1a2e1a;")
            self._detail_label.setText(f"Max deviation: {max_mm:.3f} mm — within threshold.")
        else:
            self._result_label.setText("❌  FAIL")
            self._result_label.setStyleSheet(
                "font-size:22px; font-weight:bold; padding:12px; border-radius:6px;"
                "border:2px solid #8a2a2a; color:#ff6b6b; background:#2e1a1a;")
            details = [f"Max deviation: {max_mm:.3f} mm"]
            for i, r in enumerate(fail_regions, 1):
                d = "↑ lifted" if r["direction"] == "lifted" else "↓ sunken"
                details.append(f"Region {i}: {d}  {r['max_uplift_mm']:.2f}mm  "
                                f"@ {r['x_start_mm']:.0f}–{r['x_end_mm']:.0f}mm")
            self._detail_label.setText("\n".join(details))

    def _load_baseline_silent(self):
        try:
            self._baseline_profile, data = load_baseline()
            n = data.get("num_images", 1)
            self._baseline_status.setText(
                f"Baseline loaded  ({n} image avg,  "
                f"median Y={data['baseline_median_y']:.1f}px)")
            self._baseline_status.setStyleSheet("color: #6fcf6f; font-size: 11px;")
        except FileNotFoundError:
            self._baseline_status.setText("Baseline: not loaded — calibrate first.")
            self._baseline_status.setStyleSheet("color: #ff6b6b; font-size: 11px;")

    # ── CAMERA ─────────────────────────────────
    def _start_camera(self):
        if not CAMERA_AVAILABLE:
            self._set_status("Camera not available on this machine.")
            return
        if self._cam is not None:
            return
        try:
            self._cam = Picamera2()
            preview_cfg = self._cam.create_preview_configuration(
                main={"size": (1152, 648), "format": "RGB888"})
            self._cam.configure(preview_cfg)
            self._cam.start()
            time.sleep(1)
            self._preview_thread = CameraPreviewThread(self._cam)
            self._preview_thread.frame_ready.connect(self._on_preview_frame)
            self._preview_thread.start()
            self._set_status("Camera preview started.")
        except Exception as e:
            self._set_status(f"Camera error: {e}")

    def _on_preview_frame(self, frame):
        self._preview_display.set_rgb(frame)

    def _capture_camera(self):
        if self._cam is None:
            self._set_status("Start camera preview first.")
            return
        try:
            # Switch to still config temporarily
            self._preview_thread.stop()
            self._cam.stop()
            still_cfg = self._cam.create_still_configuration(
                main={"size": CONFIG["CAPTURE_RESOLUTION"], "format": "RGB888"},
                controls={
                    "AfMode": 0,
                    "LensPosition": CONFIG["LENS_POSITION"],
                    "ExposureTime": CONFIG["EXPOSURE_TIME"],
                    "AnalogueGain": 1.0,
                    "AwbEnable": False,
                    "ColourGains": (1.5, 1.5),
                })
            self._cam.configure(still_cfg)
            self._cam.start()
            time.sleep(1)
            frame = self._cam.capture_array()
            self._cam.stop()

            # Normalize
            bgr  = normalize_image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._last_rgb = rgb
            self._captured_display.set_rgb(rgb)
            self._roi_widget.set_image(rgb)
            self._set_status("Image captured from camera.")

            # Restart preview
            preview_cfg = self._cam.create_preview_configuration(
                main={"size": (1152, 648), "format": "RGB888"})
            self._cam.configure(preview_cfg)
            self._cam.start()
            self._preview_thread = CameraPreviewThread(self._cam)
            self._preview_thread.frame_ready.connect(self._on_preview_frame)
            self._preview_thread.start()
        except Exception as e:
            self._set_status(f"Capture error: {e}")

    # ── FILE LOAD ───────────────────────────────
    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        try:
            rgb, _ = load_image_from_file(path)
            self._last_rgb = rgb
            self._captured_display.set_rgb(rgb)
            self._roi_widget.set_image(rgb)
            self._set_status(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _load_for_roi(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image for ROI", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        try:
            rgb, _ = load_image_from_file(path)
            self._last_rgb = rgb
            self._roi_widget.set_image(rgb)
            self._set_status(f"ROI image loaded: {os.path.basename(path)}")
        except Exception as e:
            self._set_status(f"Error: {e}")

    # ── ROI ─────────────────────────────────────
    def _apply_manual_roi(self):
        x1, y1 = self._roi_x1.value(), self._roi_y1.value()
        x2, y2 = self._roi_x2.value(), self._roi_y2.value()
        if x2 <= x1 or y2 <= y1:
            self._set_status("Invalid ROI: X2 must be > X1, Y2 must be > Y1.")
            return
        CONFIG["ROI_X_START"] = x1
        CONFIG["ROI_Y_START"] = y1
        CONFIG["ROI_X_END"]   = x2
        CONFIG["ROI_Y_END"]   = y2
        save_roi_config()
        self._update_roi_display()
        self._set_status(f"ROI set manually: ({x1},{y1}) → ({x2},{y2})")

    def _confirm_drawn_roi(self):
        if not self._roi_widget.confirm_roi():
            self._set_status("Draw a box on the image first, then confirm.")

    def _on_roi_drawn(self, x1, y1, x2, y2):
        CONFIG["ROI_X_START"] = x1
        CONFIG["ROI_Y_START"] = y1
        CONFIG["ROI_X_END"]   = x2
        CONFIG["ROI_Y_END"]   = y2
        save_roi_config()
        self._update_roi_display()
        self._set_status(f"ROI confirmed: ({x1},{y1}) → ({x2},{y2})  |  Saved to roi_config.json")

    # ── CALIBRATION ─────────────────────────────
    def _add_calibration_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Bare Jig Image", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        try:
            rgb, gray = load_image_from_file(path)
            profile = get_edge_profile(gray)
            self._cal_profiles.append(profile)
            self._last_rgb = rgb
            self._captured_display.set_rgb(rgb)

            n     = len(self._cal_profiles)
            total = CONFIG["CALIBRATION_IMAGES"]
            self._cal_progress.setVisible(True)
            self._cal_progress.setValue(n)
            self._cal_progress.setFormat(f"{n} / {total} images")
            self._cal_count_label.setText(
                f"{n}/{total} calibration images collected."
                + (" ← Ready to finish!" if n >= total else ""))
            self._set_status(f"Calibration image {n}/{total} added.")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _finish_calibration(self):
        if len(self._cal_profiles) == 0:
            self._set_status("Add at least 1 calibration image first.")
            return
        baseline = np.mean(self._cal_profiles, axis=0)
        data = {
            "baseline_median_y": float(np.median(baseline)),
            "baseline_per_col":  baseline.tolist(),
            "timestamp":         datetime.now().isoformat(),
            "target_width":      CONFIG["TARGET_WIDTH"],
            "num_images":        len(self._cal_profiles),
        }
        with open(CONFIG["BASELINE_FILE"], "w") as f:
            json.dump(data, f, indent=2)
        self._baseline_profile = baseline
        self._cal_profiles     = []
        self._cal_progress.setValue(0)
        self._cal_progress.setVisible(False)
        self._cal_count_label.setText("Calibration complete.")
        self._load_baseline_silent()
        self._set_status(f"Calibration saved — {data['num_images']} image average.")

    def _reset_calibration(self):
        self._cal_profiles = []
        self._cal_progress.setValue(0)
        self._cal_progress.setVisible(False)
        self._cal_count_label.setText("Calibration reset.")
        self._set_status("Calibration images cleared.")

    # ── INSPECTION ──────────────────────────────
    def _ensure_baseline(self):
        if self._baseline_profile is None:
            try:
                self._baseline_profile, _ = load_baseline()
            except FileNotFoundError:
                self._set_status("No baseline — calibrate first.")
                return False
        return True

    def _run_inspection(self, rgb, gray, label):
        if not self._ensure_baseline():
            return

        current_profile = get_edge_profile(gray)
        passed, fail_regions, diff, diff_mm, abs_diff, max_abs_mm, max_raw_mm = \
            analyze_uplift(current_profile, self._baseline_profile)

        annotated = build_annotated_image(
            rgb, current_profile, self._baseline_profile,
            abs_diff, fail_regions, passed)

        self._captured_display.set_rgb(annotated)
        self._show_result(passed, fail_regions, max_abs_mm)

        # Save log
        os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(CONFIG["LOG_DIR"],
                             f"{timestamp}_{'PASS' if passed else 'FAIL'}.jpg")
        cv2.imwrite(fname, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        entry = {
            "timestamp":    timestamp,
            "source":       label,
            "result":       "PASS" if passed else "FAIL",
            "max_mm":       round(max_abs_mm, 4),
            "fail_regions": fail_regions,
            "image":        fname,
        }
        with open(os.path.join(CONFIG["LOG_DIR"], "inspection_log.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._set_status(
            f"Inspection done — {'PASS' if passed else 'FAIL'}  "
            f"| Max: {max_abs_mm:.3f}mm  | Saved: {fname}")

    def _inspect_camera(self):
        if self._last_rgb is None:
            self._set_status("Capture an image first.")
            return
        bgr  = cv2.cvtColor(self._last_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self._run_inspection(self._last_rgb, gray, "CAMERA")

    def _inspect_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PCB Image to Inspect", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        try:
            rgb, gray = load_image_from_file(path)
            self._last_rgb = rgb
            self._run_inspection(rgb, gray, os.path.basename(path))
        except Exception as e:
            self._set_status(f"Error: {e}")

    # ── SETTINGS ────────────────────────────────
    def _apply_settings(self):
        CONFIG["UPLIFT_THRESHOLD_PX"] = self._s_threshold_px.value()
        CONFIG["MIN_FAIL_COLUMNS"]    = self._s_min_cols.value()
        CONFIG["MAX_MM_THRESHOLD"]    = self._s_max_mm.value()
        CONFIG["PX_PER_MM"]           = self._s_px_per_mm.value()
        CONFIG["CANNY_LOW"]           = self._s_canny_low.value()
        CONFIG["CANNY_HIGH"]          = self._s_canny_high.value()
        CONFIG["CALIBRATION_IMAGES"]  = self._s_cal_images.value()
        bk = self._s_blur.value()
        CONFIG["BLUR_KERNEL"]         = bk if bk % 2 == 1 else bk + 1
        self._set_status("Settings applied.")

    def closeEvent(self, e):
        if self._preview_thread:
            self._preview_thread.stop()
        if self._cam:
            try:
                self._cam.stop()
            except Exception:
                pass
        e.accept()


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
