#!/usr/bin/env python3
"""
PCB Warpage / Uplift Detection System
UI Mode: Tkinter GUI with live preview, ROI drawing, capture, calibrate, inspect
Camera: OpenCV webcam (replaces Picamera2)
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
import json
from datetime import datetime
from PIL import Image, ImageTk

# ── Webcam setup ───────────────────────────────────────────────────
CAMERA_INDEX     = 0      # Change if your webcam is not /dev/video0
CAMERA_AVAILABLE = False  # Will be set to True if webcam opens successfully

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
CONFIG = {
    "CAPTURE_RESOLUTION": (1920, 1080),   # requested from webcam (may fall back)
    "TARGET_WIDTH":   2000,
    "ROI_Y_START":    600,
    "ROI_Y_END":      900,
    "ROI_X_START":    100,
    "ROI_X_END":      1900,
    "UPLIFT_THRESHOLD_PX": 5,
    "MIN_FAIL_COLUMNS":    10,
    "MAX_MM_THRESHOLD":    0.3,
    "BASELINE_FILE":   "baseline.json",
    "ROI_CONFIG_FILE": "roi_config.json",
    "LOG_DIR":         "inspection_logs",
    "BLUR_KERNEL": 5,
    "CANNY_LOW":   30,
    "CANNY_HIGH":  100,
    "PX_PER_MM":   10.0,
}

# ──────────────────────────────────────────────
# WEBCAM HELPER
# ──────────────────────────────────────────────
class WebcamCapture:
    """
    Thin wrapper around cv2.VideoCapture.
    Opens the camera once and keeps it open for the lifetime of the app
    so frames are available instantly (no repeated open/close overhead).
    """
    def __init__(self, index: int = 0):
        self._cap   = None
        self._lock  = threading.Lock()
        self._index = index
        self._open()

    def _open(self):
        global CAMERA_AVAILABLE
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            CAMERA_AVAILABLE = False
            return
        # Request resolution — the driver will use the closest supported mode
        w, h = CONFIG["CAPTURE_RESOLUTION"]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # Give the sensor a moment to settle after first open
        time.sleep(0.5)
        self._cap        = cap
        CAMERA_AVAILABLE = True

    # ── public API ──────────────────────────────────────────────
    def read_frame(self):
        """Return a BGR numpy array, or None on failure."""
        if self._cap is None:
            return None
        with self._lock:
            ret, frame = self._cap.read()
        return frame if ret else None

    def capture_still(self, warmup_frames: int = 10):
        """
        Grab a high-quality still by discarding a few frames first
        (lets AGC / AWB converge) then returning the next frame.
        Returns BGR numpy array or None.
        """
        if self._cap is None:
            return None
        with self._lock:
            for _ in range(warmup_frames):
                self._cap.grab()            # discard
            ret, frame = self._cap.read()
        return frame if ret else None

    def stop(self):
        if self._cap:
            self._cap.release()
            self._cap = None


# ──────────────────────────────────────────────
# ROI CONFIG — load / save
# ──────────────────────────────────────────────
def load_roi_config():
    if os.path.exists(CONFIG["ROI_CONFIG_FILE"]):
        with open(CONFIG["ROI_CONFIG_FILE"]) as f:
            roi = json.load(f)
        for key in ("ROI_X_START", "ROI_X_END", "ROI_Y_START", "ROI_Y_END"):
            if key in roi:
                CONFIG[key] = roi[key]

def save_roi_config():
    roi = {k: CONFIG[k] for k in ("ROI_X_START", "ROI_X_END", "ROI_Y_START", "ROI_Y_END")}
    with open(CONFIG["ROI_CONFIG_FILE"], "w") as f:
        json.dump(roi, f, indent=2)

# ──────────────────────────────────────────────
# IMAGE NORMALISATION
# ──────────────────────────────────────────────
def normalize_image(bgr_img):
    h, w  = bgr_img.shape[:2]
    scale = CONFIG["TARGET_WIDTH"] / w
    new_h = int(h * scale)
    return cv2.resize(bgr_img, (CONFIG["TARGET_WIDTH"], new_h),
                      interpolation=cv2.INTER_AREA)

# ──────────────────────────────────────────────
# EDGE PROFILE
# ──────────────────────────────────────────────
def get_edge_profile(gray_img):
    roi = gray_img[
        CONFIG["ROI_Y_START"]:CONFIG["ROI_Y_END"],
        CONFIG["ROI_X_START"]:CONFIG["ROI_X_END"]
    ]
    blurred = cv2.GaussianBlur(roi,
                                (CONFIG["BLUR_KERNEL"], CONFIG["BLUR_KERNEL"]), 0)
    edges   = cv2.Canny(blurred, CONFIG["CANNY_LOW"], CONFIG["CANNY_HIGH"])
    roi_h   = CONFIG["ROI_Y_END"] - CONFIG["ROI_Y_START"]
    profile = np.full(edges.shape[1], fill_value=float(roi_h), dtype=np.float32)
    for col in range(edges.shape[1]):
        rows = np.where(edges[:, col] > 0)[0]
        if len(rows):
            profile[col] = rows[0]
    return profile + CONFIG["ROI_Y_START"]

# ──────────────────────────────────────────────
# UPLIFT ANALYSIS
# ──────────────────────────────────────────────
def analyze_uplift(current_profile, baseline_profile):
    if len(current_profile) != len(baseline_profile):
        baseline_profile = np.interp(
            np.linspace(0, 1, len(current_profile)),
            np.linspace(0, 1, len(baseline_profile)),
            baseline_profile,
        )
    diff     = baseline_profile - current_profile
    abs_diff = np.abs(diff)
    diff_mm  = diff     / CONFIG["PX_PER_MM"]
    abs_mm   = abs_diff / CONFIG["PX_PER_MM"]
    flagged  = abs_diff > CONFIG["UPLIFT_THRESHOLD_PX"]
    max_abs_mm = float(np.max(abs_mm))
    max_raw_mm = float(diff_mm[np.argmax(abs_diff)])

    fail_regions = []
    in_region = False; region_start = 0; total_cols = len(flagged)
    for i in range(total_cols + 1):
        is_flag = (i < total_cols) and flagged[i]
        if is_flag and not in_region:
            in_region = True; region_start = i
        elif not is_flag and in_region:
            in_region = False; length = i - region_start
            if length >= CONFIG["MIN_FAIL_COLUMNS"]:
                seg    = diff[region_start:i]
                seg_mm = diff_mm[region_start:i]
                fail_regions.append({
                    "col_start":     region_start, "col_end": i,
                    "max_uplift_px": float(np.max(np.abs(seg))),
                    "max_uplift_mm": float(np.max(np.abs(seg_mm))),
                    "direction":     "lifted" if np.mean(seg) > 0 else "sunken",
                    "x_start_mm":   (region_start / total_cols) * 200.0,
                    "x_end_mm":     (i            / total_cols) * 200.0,
                })
    passed = not (len(fail_regions) > 0 or max_abs_mm > CONFIG["MAX_MM_THRESHOLD"])
    return passed, fail_regions, diff, diff_mm, max_abs_mm, max_raw_mm

# ──────────────────────────────────────────────
# ANNOTATED IMAGE
# ──────────────────────────────────────────────
def build_annotated_image(color_bgr, current_profile, baseline_profile,
                           diff, fail_regions, passed):
    ann      = color_bgr.copy()
    roi_x0   = CONFIG["ROI_X_START"]
    abs_diff = np.abs(diff)

    for col_idx, by in enumerate(baseline_profile):
        x, y = col_idx + roi_x0, int(by)
        if 0 <= y < ann.shape[0] and 0 <= x < ann.shape[1]:
            ann[y, x] = (0, 255, 0)

    for col_idx, cy in enumerate(current_profile):
        x, y  = col_idx + roi_x0, int(cy)
        color = (0, 0, 255) if abs_diff[col_idx] > CONFIG["UPLIFT_THRESHOLD_PX"] \
                else (0, 255, 255)
        if 0 <= y < ann.shape[0] and 0 <= x < ann.shape[1]:
            ann[y, x] = color

    cv2.rectangle(ann,
                  (CONFIG["ROI_X_START"], CONFIG["ROI_Y_START"]),
                  (CONFIG["ROI_X_END"],   CONFIG["ROI_Y_END"]),
                  (0, 165, 255), 2)

    label = "PASS" if passed else f"FAIL — {len(fail_regions)} region(s)"
    color = (0, 200, 0) if passed else (0, 0, 255)
    cv2.putText(ann, label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5)

    y_off = 160
    for r in fail_regions:
        d    = "lifted" if r["direction"] == "lifted" else "sunken"
        text = (f"  {d}  {r['max_uplift_mm']:.2f}mm "
                f"@ {r['x_start_mm']:.0f}–{r['x_end_mm']:.0f}mm")
        cv2.putText(ann, text, (50, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        y_off += 55
    return ann

# ──────────────────────────────────────────────
# BASELINE LOAD / SAVE
# ──────────────────────────────────────────────
def load_baseline():
    if not os.path.exists(CONFIG["BASELINE_FILE"]):
        raise FileNotFoundError("No baseline.json — run Calibrate first.")
    with open(CONFIG["BASELINE_FILE"]) as f:
        data = json.load(f)
    return np.array(data["baseline_per_col"])

def save_baseline(profile, bgr_img):
    data = {
        "baseline_median_y": float(np.median(profile)),
        "baseline_per_col":  profile.tolist(),
        "timestamp":         datetime.now().isoformat(),
        "target_width":      CONFIG["TARGET_WIDTH"],
    }
    with open(CONFIG["BASELINE_FILE"], "w") as f:
        json.dump(data, f, indent=2)
    os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
    cv2.imwrite(
        os.path.join(CONFIG["LOG_DIR"], "calibration_image.jpg"),
        bgr_img,
    )

# ──────────────────────────────────────────────
# LOG ENTRY
# ──────────────────────────────────────────────
def write_log(timestamp, source, passed, max_abs_mm, fail_regions, img_path):
    os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
    entry = {
        "timestamp":    timestamp,
        "source":       source,
        "result":       "PASS" if passed else "FAIL",
        "max_mm":       round(max_abs_mm, 4),
        "fail_regions": fail_regions,
        "image":        img_path,
    }
    with open(os.path.join(CONFIG["LOG_DIR"], "inspection_log.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")

# ═══════════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════════
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
BORDER     = "#30363d"
TEXT_FG    = "#e6edf3"
MUTED      = "#8b949e"
ACCENT     = "#58a6ff"
GREEN      = "#3fb950"
RED        = "#f85149"
ORANGE     = "#d29922"
BTN_BG     = "#21262d"
BTN_HOV    = "#30363d"
FONT_MONO  = ("Courier New", 11)
FONT_LABEL = ("Segoe UI", 10)
FONT_TITLE = ("Segoe UI", 11, "bold")


def pil_from_bgr(bgr, max_w, max_h):
    """Resize BGR numpy array to fit (max_w × max_h), return ImageTk.PhotoImage."""
    h, w  = bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    small  = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


class PCBApp(tk.Tk):
    PREVIEW_W = 760
    PREVIEW_H = 430
    RESULT_W  = 760
    RESULT_H  = 430

    def __init__(self):
        super().__init__()
        self.title("PCB Warpage Detection System")
        self.configure(bg=DARK_BG)
        self.resizable(True, True)

        # State
        self.cam              = None
        self.baseline_profile = None
        self.last_bgr         = None    # most recent captured BGR frame
        self.last_gray        = None
        self._preview_running = False
        self._preview_thread  = None
        self._live_frame_bgr  = None
        self._live_lock       = threading.Lock()

        # Live-inspect mode
        self._live_inspect_active = False
        self._live_result         = None
        self._live_result_detail  = ""
        self._live_inspect_every  = 10
        self._live_frame_count    = 0

        # ROI drawing state
        self._roi_drawing  = False
        self._roi_start    = None
        self._roi_rect_id  = None
        self._roi_mode     = False
        self._result_scale  = 1.0
        self._result_offset = (0, 0)

        load_roi_config()
        self._build_ui()
        self._try_load_baseline()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Try to open the webcam
        self._start_camera()

    # ── UI CONSTRUCTION ───────────────────────────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── HEADER ──
        hdr = tk.Frame(self, bg=PANEL_BG, bd=0)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.columnconfigure(1, weight=1)

        tk.Label(hdr, text="⬡  PCB WARPAGE INSPECTOR",
                 font=("Segoe UI", 14, "bold"),
                 bg=PANEL_BG, fg=ACCENT, padx=16, pady=10).grid(row=0, column=0, sticky="w")

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(hdr, textvariable=self._status_var,
                 font=FONT_LABEL, bg=PANEL_BG, fg=MUTED, padx=16).grid(
            row=0, column=1, sticky="e")

        tk.Frame(self, bg=BORDER, height=1).grid(row=0, column=0, sticky="ews")

        # ── MAIN BODY ──
        body = tk.Frame(self, bg=DARK_BG)
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=10)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        # ── LEFT: Two image panels ──
        panels = tk.Frame(body, bg=DARK_BG)
        panels.grid(row=0, column=0, sticky="nsew")
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)
        panels.rowconfigure(0, weight=0)
        panels.rowconfigure(1, weight=1)

        tk.Label(panels, text="LIVE PREVIEW", font=FONT_TITLE,
                 bg=DARK_BG, fg=MUTED).grid(row=0, column=0, pady=(0, 4), sticky="w", padx=4)
        tk.Label(panels, text="CAPTURE / RESULT", font=FONT_TITLE,
                 bg=DARK_BG, fg=MUTED).grid(row=0, column=1, pady=(0, 4), sticky="w", padx=4)

        self._live_canvas = tk.Canvas(
            panels, width=self.PREVIEW_W, height=self.PREVIEW_H,
            bg="#0a0e14", highlightthickness=1, highlightbackground=BORDER,
        )
        self._live_canvas.grid(row=1, column=0, padx=(0, 6), sticky="nsew")
        self._live_canvas.create_text(
            self.PREVIEW_W // 2, self.PREVIEW_H // 2,
            text="Waiting for webcam…", fill=MUTED, font=FONT_MONO,
            tags="placeholder",
        )

        self._result_canvas = tk.Canvas(
            panels, width=self.RESULT_W, height=self.RESULT_H,
            bg="#0a0e14", highlightthickness=1, highlightbackground=BORDER,
            cursor="crosshair",
        )
        self._result_canvas.grid(row=1, column=1, sticky="nsew")
        self._result_canvas.create_text(
            self.RESULT_W // 2, self.RESULT_H // 2,
            text="Captured image appears here", fill=MUTED, font=FONT_MONO,
            tags="placeholder",
        )
        self._result_canvas.bind("<ButtonPress-1>",   self._roi_mouse_down)
        self._result_canvas.bind("<B1-Motion>",       self._roi_mouse_move)
        self._result_canvas.bind("<ButtonRelease-1>", self._roi_mouse_up)

        # ── RIGHT: Sidebar ──
        side = tk.Frame(body, bg=PANEL_BG, width=230,
                        highlightthickness=1, highlightbackground=BORDER)
        side.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        side.columnconfigure(0, weight=1)
        side.grid_propagate(False)
        self._build_sidebar(side)

        # ── BOTTOM: LOG ──
        log_frame = tk.Frame(self, bg=PANEL_BG,
                             highlightthickness=1, highlightbackground=BORDER)
        log_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)

        tk.Label(log_frame, text="ACTIVITY LOG", font=FONT_TITLE,
                 bg=PANEL_BG, fg=MUTED, padx=8, pady=4).grid(row=0, column=0, sticky="w")

        self._log_text = tk.Text(
            log_frame, height=7, bg="#0d1117", fg=TEXT_FG,
            font=FONT_MONO, relief="flat", insertbackground=ACCENT,
            selectbackground=BORDER, wrap="word", state="disabled",
        )
        self._log_text.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))

        sb = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        sb.grid(row=1, column=1, sticky="ns", pady=(0, 6))
        self._log_text["yscrollcommand"] = sb.set

    def _build_sidebar(self, parent):
        pad = {"padx": 12, "pady": 5}

        tk.Label(parent, text="CONTROLS", font=("Segoe UI", 9, "bold"),
                 bg=PANEL_BG, fg=MUTED).grid(row=0, column=0, sticky="w",
                                              padx=12, pady=(14, 2))

        self._btn_capture = self._make_btn(
            parent, "📷  Capture Image", self._action_capture, row=1, **pad)

        tk.Frame(parent, bg=BORDER, height=1).grid(
            row=2, column=0, sticky="ew", padx=10, pady=6)

        tk.Label(parent, text="CALIBRATION", font=("Segoe UI", 9, "bold"),
                 bg=PANEL_BG, fg=MUTED).grid(row=3, column=0, sticky="w",
                                              padx=12, pady=(0, 2))

        self._btn_set_roi = self._make_btn(
            parent, "✏  Draw ROI", self._action_set_roi, row=4, **pad)
        self._btn_calibrate = self._make_btn(
            parent, "⚙  Calibrate (Set Reference)", self._action_calibrate, row=5, **pad)

        tk.Frame(parent, bg=BORDER, height=1).grid(
            row=6, column=0, sticky="ew", padx=10, pady=6)

        tk.Label(parent, text="INSPECTION", font=("Segoe UI", 9, "bold"),
                 bg=PANEL_BG, fg=MUTED).grid(row=7, column=0, sticky="w",
                                              padx=12, pady=(0, 2))

        self._btn_inspect = self._make_btn(
            parent, "🔍  Inspect Captured Image", self._action_inspect, row=8, **pad)

        self._btn_live_inspect = self._make_btn(
            parent, "▶  Start Live Inspect", self._action_toggle_live_inspect,
            row=9, **pad)
        self._btn_live_inspect.configure(fg=GREEN)

        tk.Frame(parent, bg=BORDER, height=1).grid(
            row=10, column=0, sticky="ew", padx=10, pady=6)

        tk.Label(parent, text="CURRENT ROI", font=("Segoe UI", 9, "bold"),
                 bg=PANEL_BG, fg=MUTED).grid(row=11, column=0, sticky="w",
                                              padx=12, pady=(0, 2))

        self._roi_var = tk.StringVar()
        self._update_roi_label()
        tk.Label(parent, textvariable=self._roi_var, font=("Courier New", 9),
                 bg=PANEL_BG, fg=TEXT_FG, justify="left").grid(
            row=12, column=0, sticky="w", padx=12, pady=(0, 6))

        tk.Frame(parent, bg=BORDER, height=1).grid(
            row=13, column=0, sticky="ew", padx=10, pady=6)

        tk.Label(parent, text="BASELINE", font=("Segoe UI", 9, "bold"),
                 bg=PANEL_BG, fg=MUTED).grid(row=14, column=0, sticky="w",
                                              padx=12, pady=(0, 2))

        self._baseline_var = tk.StringVar(value="Not loaded")
        self._baseline_lbl = tk.Label(parent, textvariable=self._baseline_var,
                                      font=FONT_MONO, bg=PANEL_BG, fg=ORANGE,
                                      wraplength=200)
        self._baseline_lbl.grid(row=15, column=0, sticky="w", padx=12, pady=(0, 10))

        self._result_var = tk.StringVar(value="")
        self._result_banner = tk.Label(
            parent, textvariable=self._result_var,
            font=("Segoe UI", 20, "bold"), bg=PANEL_BG, fg=MUTED,
            relief="flat", pady=10,
        )
        self._result_banner.grid(row=16, column=0, sticky="ew", padx=12, pady=4)

        self._detail_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._detail_var, font=("Courier New", 9),
                 bg=PANEL_BG, fg=TEXT_FG, wraplength=205, justify="left").grid(
            row=17, column=0, sticky="w", padx=12)

    def _make_btn(self, parent, text, cmd, row, **grid_kw):
        btn = tk.Button(
            parent, text=text, command=cmd,
            bg=BTN_BG, fg=TEXT_FG, activebackground=BTN_HOV,
            activeforeground=TEXT_FG,
            relief="flat", bd=0, padx=10, pady=7,
            font=("Segoe UI", 10), anchor="w", width=22, cursor="hand2",
        )
        btn.grid(row=row, column=0, sticky="ew", **grid_kw)
        btn.bind("<Enter>", lambda e: btn.configure(bg=BTN_HOV))
        btn.bind("<Leave>", lambda e: btn.configure(bg=BTN_BG))
        return btn

    # ── LOGGING ──────────────────────────────────────────────────
    def _log(self, msg):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {msg}\n"
        self._log_text.configure(state="normal")
        self._log_text.insert("end", line)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _set_status(self, msg):
        self._status_var.set(msg)

    # ── BASELINE ─────────────────────────────────────────────────
    def _try_load_baseline(self):
        try:
            self.baseline_profile = load_baseline()
            self._baseline_var.set("✓ Loaded")
            self._baseline_lbl.configure(fg=GREEN)
            self._log("Baseline loaded from baseline.json")
        except FileNotFoundError:
            self._baseline_var.set("Not calibrated")
            self._baseline_lbl.configure(fg=ORANGE)

    def _update_roi_label(self):
        self._roi_var.set(
            f"X: {CONFIG['ROI_X_START']} → {CONFIG['ROI_X_END']}\n"
            f"Y: {CONFIG['ROI_Y_START']} → {CONFIG['ROI_Y_END']}"
        )

    # ── CAMERA ───────────────────────────────────────────────────
    def _start_camera(self):
        """Open the webcam. Falls back to file-only mode if unavailable."""
        try:
            self.cam = WebcamCapture(CAMERA_INDEX)
            if not CAMERA_AVAILABLE:
                raise RuntimeError("Could not open webcam.")
            actual_w = int(self.cam._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cam._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._preview_running = True
            self._preview_thread  = threading.Thread(
                target=self._preview_loop, daemon=True)
            self._preview_thread.start()
            self._log(f"Webcam started ({actual_w}×{actual_h}). Live preview active.")
            self._set_status("Camera live")
        except Exception as e:
            self._log(f"⚠  Camera error: {e}  — file upload mode only.")
            self._set_status("No camera")
            self.cam = None

    def _preview_loop(self):
        """
        Background thread — reads webcam frames continuously.
        When live-inspect is active, runs full analysis every N frames
        and overlays a coloured PASS/FAIL verdict on the preview.
        """
        while self._preview_running:
            try:
                frame = self.cam.read_frame()   # BGR
                if frame is None:
                    time.sleep(0.05)
                    continue

                bgr    = normalize_image(frame)
                bgr_ov = bgr.copy()

                # ── Live inspection (throttled) ─────────────────────
                if self._live_inspect_active and self.baseline_profile is not None:
                    self._live_frame_count += 1
                    if self._live_frame_count >= self._live_inspect_every:
                        self._live_frame_count = 0
                        try:
                            gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                            profile = get_edge_profile(gray)
                            passed, fail_regions, diff, diff_mm, max_abs_mm, _ = \
                                analyze_uplift(profile, self.baseline_profile)
                            self._live_result = "PASS" if passed else "FAIL"
                            if passed:
                                self._live_result_detail = f"max {max_abs_mm:.3f}mm"
                            else:
                                dirs = [("↑" if r["direction"] == "lifted" else "↓")
                                        for r in fail_regions]
                                self._live_result_detail = (
                                    f"max {max_abs_mm:.3f}mm  "
                                    + "  ".join(
                                        f"{d}{r['max_uplift_mm']:.2f}mm"
                                        for d, r in zip(dirs, fail_regions)
                                    )
                                )
                            self.after(0, self._update_live_result_banner)
                        except Exception:
                            pass

                # ── ROI box overlay ─────────────────────────────────
                roi_color = (0, 165, 255)
                if self._live_inspect_active and self._live_result == "PASS":
                    roi_color = (0, 220, 0)
                elif self._live_inspect_active and self._live_result == "FAIL":
                    roi_color = (0, 0, 255)

                cv2.rectangle(bgr_ov,
                              (CONFIG["ROI_X_START"], CONFIG["ROI_Y_START"]),
                              (CONFIG["ROI_X_END"],   CONFIG["ROI_Y_END"]),
                              roi_color, 3)

                # ── PASS/FAIL text on preview ────────────────────────
                if self._live_inspect_active and self._live_result:
                    verdict     = self._live_result
                    vcolor      = (0, 220, 0) if verdict == "PASS" else (0, 0, 255)
                    text_y_main = max(CONFIG["ROI_Y_START"] - 18, 60)
                    cv2.putText(bgr_ov, verdict,
                                (CONFIG["ROI_X_START"], text_y_main),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.2, vcolor, 5)
                    cv2.putText(bgr_ov, self._live_result_detail,
                                (CONFIG["ROI_X_START"], text_y_main + 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, vcolor, 2)

                with self._live_lock:
                    self._live_frame_bgr = bgr_ov
                self.after(0, self._refresh_preview)

            except Exception:
                pass
            time.sleep(0.04)   # ~25 fps

    def _refresh_preview(self):
        with self._live_lock:
            frame = self._live_frame_bgr
        if frame is None:
            return
        photo = pil_from_bgr(frame, self.PREVIEW_W, self.PREVIEW_H)
        self._live_canvas.delete("all")
        cw = self._live_canvas.winfo_width()  or self.PREVIEW_W
        ch = self._live_canvas.winfo_height() or self.PREVIEW_H
        self._live_canvas.create_image(cw // 2, ch // 2, anchor="center",
                                       image=photo, tags="frame")
        self._live_canvas._photo = photo

    # ── LIVE INSPECT TOGGLE ───────────────────────────────────────
    def _action_toggle_live_inspect(self):
        if not CAMERA_AVAILABLE or self.cam is None:
            messagebox.showinfo("No Camera",
                                "Live Inspect requires a connected webcam.")
            return
        if self.baseline_profile is None:
            messagebox.showinfo("No Baseline",
                                "Calibrate first before starting Live Inspect.")
            return

        self._live_inspect_active = not self._live_inspect_active

        if self._live_inspect_active:
            self._live_result        = None
            self._live_result_detail = ""
            self._live_frame_count   = 0
            self._btn_live_inspect.configure(text="⏹  Stop Live Inspect", fg=RED)
            self._log("Live Inspect started — classifying every frame in ROI.")
            self._set_status("Live Inspect ON")
            self._result_var.set("")
            self._detail_var.set("")
        else:
            self._btn_live_inspect.configure(text="▶  Start Live Inspect", fg=GREEN)
            self._live_result        = None
            self._live_result_detail = ""
            self._result_var.set("")
            self._detail_var.set("")
            self._log("Live Inspect stopped.")
            self._set_status("Live Inspect OFF")

    def _update_live_result_banner(self):
        if not self._live_inspect_active or self._live_result is None:
            return
        if self._live_result == "PASS":
            self._result_var.set("✅  PASS")
            self._result_banner.configure(fg=GREEN)
        else:
            self._result_var.set("❌  FAIL")
            self._result_banner.configure(fg=RED)
        self._detail_var.set(self._live_result_detail)

    # ── CAPTURE ──────────────────────────────────────────────────
    def _action_capture(self):
        if self.cam:
            self._capture_from_camera()
        else:
            self._capture_from_file()

    def _capture_from_camera(self):
        """
        Grab a still from the webcam.
        Uses capture_still() which discards a few frames first so AGC/AWB
        can settle, giving a cleaner image than a raw read_frame().
        """
        self._set_status("Capturing…")
        self._log("Capturing still from webcam…")
        try:
            frame = self.cam.capture_still(warmup_frames=10)
            if frame is None:
                raise RuntimeError("Webcam returned no frame.")
            bgr  = normalize_image(frame)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            self.last_bgr  = bgr
            self.last_gray = gray
            self._show_captured(bgr)
            self._log(f"Captured. Size: {bgr.shape[1]}×{bgr.shape[0]} px")
            self._set_status("Image captured")
        except Exception as e:
            self._log(f"⚠  Capture error: {e}")
            self._set_status("Capture failed")

    def _capture_from_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select PCB image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not path:
            return
        try:
            bgr  = cv2.imread(path)
            if bgr is None:
                raise RuntimeError(f"Could not read: {path}")
            bgr  = normalize_image(bgr)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            self.last_bgr  = bgr
            self.last_gray = gray
            self._show_captured(bgr)
            self._log(f"Loaded file: {os.path.basename(path)}")
            self._set_status("File loaded")
        except Exception as e:
            self._log(f"⚠  File load error: {e}")

    def _show_captured(self, bgr):
        cw = self._result_canvas.winfo_width()  or self.RESULT_W
        ch = self._result_canvas.winfo_height() or self.RESULT_H
        h, w = bgr.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        x_off = (cw - nw) // 2
        y_off = (ch - nh) // 2
        self._result_scale  = scale
        self._result_offset = (x_off, y_off)

        photo = pil_from_bgr(bgr, cw, ch)
        self._result_canvas.delete("all")
        self._result_canvas.create_image(cw // 2, ch // 2, anchor="center",
                                         image=photo, tags="captured")
        self._result_canvas._photo = photo
        self._draw_roi_on_canvas()

    def _draw_roi_on_canvas(self):
        self._result_canvas.delete("roi_box")
        sc = self._result_scale
        ox, oy = self._result_offset
        x1 = int(CONFIG["ROI_X_START"] * sc) + ox
        y1 = int(CONFIG["ROI_Y_START"] * sc) + oy
        x2 = int(CONFIG["ROI_X_END"]   * sc) + ox
        y2 = int(CONFIG["ROI_Y_END"]   * sc) + oy
        self._result_canvas.create_rectangle(
            x1, y1, x2, y2,
            outline=ORANGE, width=2, tags="roi_box", dash=(6, 4),
        )

    # ── ROI DRAWING ──────────────────────────────────────────────
    def _action_set_roi(self):
        if self.last_bgr is None:
            messagebox.showinfo("No Image",
                                "Capture or load an image first, then draw ROI.")
            return
        self._roi_mode = True
        self._result_canvas.configure(cursor="crosshair")
        self._log("ROI mode ON — click and drag on the result image to draw ROI.")
        self._set_status("Draw ROI on captured image…")

    def _roi_mouse_down(self, event):
        if not self._roi_mode:
            return
        self._roi_drawing = True
        self._roi_start   = (event.x, event.y)
        if self._roi_rect_id:
            self._result_canvas.delete(self._roi_rect_id)

    def _roi_mouse_move(self, event):
        if not self._roi_mode or not self._roi_drawing:
            return
        if self._roi_rect_id:
            self._result_canvas.delete(self._roi_rect_id)
        x0, y0 = self._roi_start
        self._roi_rect_id = self._result_canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline=GREEN, width=2, dash=(4, 3),
        )

    def _roi_mouse_up(self, event):
        if not self._roi_mode or not self._roi_drawing:
            return
        self._roi_drawing = False
        self._roi_mode    = False
        self._result_canvas.configure(cursor="crosshair")

        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        sc = self._result_scale
        ox, oy = self._result_offset
        img_x0 = int((min(x0, x1) - ox) / sc)
        img_y0 = int((min(y0, y1) - oy) / sc)
        img_x1 = int((max(x0, x1) - ox) / sc)
        img_y1 = int((max(y0, y1) - oy) / sc)

        H, W = self.last_bgr.shape[:2]
        img_x0 = max(0, min(img_x0, W))
        img_y0 = max(0, min(img_y0, H))
        img_x1 = max(0, min(img_x1, W))
        img_y1 = max(0, min(img_y1, H))

        if abs(img_x1 - img_x0) < 20 or abs(img_y1 - img_y0) < 5:
            self._log("⚠  ROI too small — try again.")
            self._set_status("ROI too small")
            return

        CONFIG["ROI_X_START"] = img_x0
        CONFIG["ROI_Y_START"] = img_y0
        CONFIG["ROI_X_END"]   = img_x1
        CONFIG["ROI_Y_END"]   = img_y1
        save_roi_config()
        self._update_roi_label()
        self._draw_roi_on_canvas()
        self._log(f"ROI saved: X {img_x0}→{img_x1}  Y {img_y0}→{img_y1}")
        self._set_status("ROI updated")

    # ── CALIBRATE ────────────────────────────────────────────────
    def _action_calibrate(self):
        if self.last_bgr is None or self.last_gray is None:
            messagebox.showinfo("No Image",
                                "Capture or load the BARE JIG image first, then calibrate.")
            return
        ans = messagebox.askyesno(
            "Calibrate",
            "This will overwrite the existing baseline.\n"
            "Make sure the captured image shows the bare jig (no PCB).\n\nContinue?",
        )
        if not ans:
            return
        try:
            profile = get_edge_profile(self.last_gray)
            save_baseline(profile, self.last_bgr)
            self.baseline_profile = profile
            self._baseline_var.set("✓ Calibrated now")
            self._baseline_lbl.configure(fg=GREEN)
            self._log(f"Calibration done. Baseline median Y = {np.median(profile):.1f} px")
            self._set_status("Calibrated")
            self._result_var.set("")
            self._detail_var.set("")
        except Exception as e:
            self._log(f"⚠  Calibration error: {e}")

    # ── INSPECT ──────────────────────────────────────────────────
    def _action_inspect(self):
        if self.last_gray is None:
            messagebox.showinfo("No Image", "Capture or load a PCB image first.")
            return
        if self.baseline_profile is None:
            messagebox.showinfo("No Baseline",
                                "No baseline loaded. Run Calibrate first.")
            return

        self._set_status("Inspecting…")
        self._log("Running inspection…")

        try:
            current_profile = get_edge_profile(self.last_gray)
            passed, fail_regions, diff, diff_mm, max_abs_mm, max_raw_mm = analyze_uplift(
                current_profile, self.baseline_profile
            )

            bgr = self.last_bgr
            ann = build_annotated_image(
                bgr, current_profile, self.baseline_profile,
                diff, fail_regions, passed,
            )

            cw = self._result_canvas.winfo_width()  or self.RESULT_W
            ch = self._result_canvas.winfo_height() or self.RESULT_H
            photo = pil_from_bgr(ann, cw, ch)
            self._result_canvas.delete("all")
            self._result_canvas.create_image(cw // 2, ch // 2, anchor="center",
                                             image=photo, tags="result")
            self._result_canvas._photo = photo

            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            label = "PASS" if passed else "FAIL"
            os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
            img_path = os.path.join(CONFIG["LOG_DIR"], f"{ts}_{label}.jpg")
            cv2.imwrite(img_path, ann)
            write_log(ts, "UI", passed, max_abs_mm, fail_regions, img_path)

            if passed:
                self._result_var.set("✅  PASS")
                self._result_banner.configure(fg=GREEN)
                self._detail_var.set(
                    f"Max deviation: {max_abs_mm:.3f} mm\n"
                    f"Threshold: {CONFIG['MAX_MM_THRESHOLD']} mm"
                )
            else:
                self._result_var.set("❌  FAIL")
                self._result_banner.configure(fg=RED)
                details = [f"Max deviation: {max_abs_mm:.3f} mm"]
                for i, r in enumerate(fail_regions, 1):
                    d = "↑ lifted" if r["direction"] == "lifted" else "↓ sunken"
                    details.append(
                        f"  {i}. {d}  {r['max_uplift_mm']:.2f}mm "
                        f"@ {r['x_start_mm']:.0f}–{r['x_end_mm']:.0f}mm"
                    )
                self._detail_var.set("\n".join(details))

            self._log(f"Result: {label}  |  max={max_abs_mm:.3f}mm  |  saved → {img_path}")
            self._set_status(label)

        except Exception as e:
            self._log(f"⚠  Inspection error: {e}")
            self._set_status("Error")

    # ── CLOSE ────────────────────────────────────────────────────
    def _on_close(self):
        self._preview_running = False
        if self.cam:
            try:
                self.cam.stop()
            except Exception:
                pass
        self.destroy()


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = PCBApp()
    app.mainloop()
