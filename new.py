#!/usr/bin/env python3
"""
PCB Warpage / Uplift Detection System
Mode: Manual capture + file upload support + interactive ROI selector
"""

import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
from tkinter import Tk, filedialog

try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("[WARN] picamera2 not found — camera options disabled. File mode only.")

# ──────────────────────────────────────────────
# CONFIGURATION
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

    "UPLIFT_THRESHOLD_PX": 6,     # CHANGE 6: raised from 5 → safer
    "MIN_FAIL_COLUMNS":    10,
    "MAX_MM_THRESHOLD":    0.3,

    "BASELINE_FILE":   "baseline.json",
    "ROI_CONFIG_FILE": "roi_config.json",
    "LOG_DIR":         "inspection_logs",

    "BLUR_KERNEL": 5,
    "CANNY_LOW":   30,
    "CANNY_HIGH":  100,

    "PX_PER_MM": 10.0,

    "CALIBRATION_IMAGES": 5,      # CHANGE 5: how many images to average for baseline
}

# ──────────────────────────────────────────────
# FILE PICKER
# ──────────────────────────────────────────────
def pick_image_file(title="Select Image"):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return path

# ──────────────────────────────────────────────
# ROI CONFIG
# ──────────────────────────────────────────────
def load_roi_config():
    if os.path.exists(CONFIG["ROI_CONFIG_FILE"]):
        with open(CONFIG["ROI_CONFIG_FILE"]) as f:
            roi = json.load(f)
        for key in ("ROI_X_START", "ROI_X_END", "ROI_Y_START", "ROI_Y_END"):
            if key in roi:
                CONFIG[key] = roi[key]
        print(f"[ROI] Loaded: "
              f"X={CONFIG['ROI_X_START']}–{CONFIG['ROI_X_END']}, "
              f"Y={CONFIG['ROI_Y_START']}–{CONFIG['ROI_Y_END']}")
    else:
        print("[ROI] No roi_config.json found — using defaults. Run option 4 to set ROI.")

def save_roi_config():
    roi = {
        "ROI_X_START": CONFIG["ROI_X_START"],
        "ROI_X_END":   CONFIG["ROI_X_END"],
        "ROI_Y_START": CONFIG["ROI_Y_START"],
        "ROI_Y_END":   CONFIG["ROI_Y_END"],
    }
    with open(CONFIG["ROI_CONFIG_FILE"], "w") as f:
        json.dump(roi, f, indent=2)

# ──────────────────────────────────────────────
# IMAGE LOADING + NORMALIZATION
# ──────────────────────────────────────────────
def normalize_image(bgr_img):
    h, w = bgr_img.shape[:2]
    scale = CONFIG["TARGET_WIDTH"] / w
    resized = cv2.resize(bgr_img,
                         (CONFIG["TARGET_WIDTH"], int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return resized

def load_image_from_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    bgr  = normalize_image(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"[FILE] Loaded: {os.path.basename(path)}  →  {bgr.shape[1]}×{bgr.shape[0]} px")
    return rgb, gray

# ──────────────────────────────────────────────
# CAMERA
# ──────────────────────────────────────────────
def init_camera():
    if not CAMERA_AVAILABLE:
        print("[ERROR] Camera not available on this machine.")
        return None
    cam = Picamera2()
    cfg = cam.create_still_configuration(
        main={"size": CONFIG["CAPTURE_RESOLUTION"], "format": "RGB888"},
        controls={
            "AfMode": 0,
            "LensPosition": CONFIG["LENS_POSITION"],
            "ExposureTime": CONFIG["EXPOSURE_TIME"],
            "AnalogueGain": 1.0,
            "AwbEnable": False,
            "ColourGains": (1.5, 1.5),
        }
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(2)
    print("[CAM] Camera ready. Focus locked.")
    return cam

def capture_from_camera(cam):
    frame = cam.capture_array()
    bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bgr   = normalize_image(bgr)
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, gray

# ──────────────────────────────────────────────
# IMAGE SOURCE PICKER
# ──────────────────────────────────────────────
def get_image(prompt, cam=None):
    print(f"\n[{prompt}] Choose image source:")
    if CAMERA_AVAILABLE and cam:
        print("  1. Capture from camera")
        print("  2. Pick from file (dialog)")
    else:
        print("  1. Pick from file (dialog)  [camera not available]")

    choice = input("  Choice: ").strip()

    if choice == "1" and CAMERA_AVAILABLE and cam:
        input("  Press Enter to capture...")
        return capture_from_camera(cam)
    else:
        print("  [FILE PICKER] Opening dialog...")
        path = pick_image_file(title=prompt)
        if not path:
            print("  [CANCELLED]")
            return None, None
        try:
            return load_image_from_file(path)
        except Exception as e:
            print(f"  [ERROR] {e}")
            return None, None

# ──────────────────────────────────────────────
# INTERACTIVE ROI SELECTOR
# ──────────────────────────────────────────────
def select_roi_interactively(image_rgb):
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    max_display = 1280
    scale = min(max_display / w, max_display / h, 1.0)
    display_base = cv2.resize(bgr, (int(w * scale), int(h * scale)))

    start_pt = None
    end_pt   = None
    drawing  = False

    def draw_overlay():
        out = display_base.copy()
        cv2.putText(out,
                    "Draw box over jig top-edge strip  |  Enter=confirm  C=clear  Q=cancel",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        if start_pt and end_pt:
            cv2.rectangle(out, start_pt, end_pt, (0, 255, 0), 2)
            bw = abs(end_pt[0] - start_pt[0])
            bh = abs(end_pt[1] - start_pt[1])
            cv2.putText(out, f"{bw}x{bh}px",
                        (min(start_pt[0], end_pt[0]),
                         min(start_pt[1], end_pt[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return out

    def mouse(event, x, y, flags, param):
        nonlocal start_pt, end_pt, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing  = True
            start_pt = (x, y)
            end_pt   = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_pt  = (x, y)

    print("\n[ROI] Window opening.")
    print("      Draw a wide horizontal strip over the PCB top edge.")
    print("      Leave ~1.5cm of empty space ABOVE the edge inside the box.")
    print("      Width = full 20cm board. Press Enter to confirm.\n")

    cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Selector", mouse)

    result = None
    while True:
        cv2.imshow("ROI Selector", draw_overlay())
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):
            if start_pt and end_pt and start_pt != end_pt:
                x1 = int(min(start_pt[0], end_pt[0]) / scale)
                y1 = int(min(start_pt[1], end_pt[1]) / scale)
                x2 = int(max(start_pt[0], end_pt[0]) / scale)
                y2 = int(max(start_pt[1], end_pt[1]) / scale)
                result = (x1, y1, x2, y2)
                break
            else:
                print("[ROI] No box drawn yet.")
        elif key in (ord('c'), ord('C')):
            start_pt = end_pt = None
            print("[ROI] Cleared. Draw again.")
        elif key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()
    return result

def run_roi_setup(cam=None):
    rgb, _ = get_image("SET ROI", cam)
    if rgb is None:
        return

    result = select_roi_interactively(rgb)
    if result is None:
        print("[ROI] Cancelled.")
        return

    x1, y1, x2, y2 = result
    CONFIG["ROI_X_START"] = x1
    CONFIG["ROI_Y_START"] = y1
    CONFIG["ROI_X_END"]   = x2
    CONFIG["ROI_Y_END"]   = y2
    save_roi_config()

    print(f"\n[ROI] Saved!")
    print(f"      X: {x1} → {x2}  ({x2-x1} px wide)")
    print(f"      Y: {y1} → {y2}  ({y2-y1} px tall)")

# ──────────────────────────────────────────────
# EDGE PROFILE EXTRACTION
# CHANGE 1: median of first 5 edge pixels (robust against noise)
# CHANGE 2: profile smoothed with Gaussian after extraction
# ──────────────────────────────────────────────
def get_edge_profile(gray_img):
    roi = gray_img[
        CONFIG["ROI_Y_START"]:CONFIG["ROI_Y_END"],
        CONFIG["ROI_X_START"]:CONFIG["ROI_X_END"]
    ]
    blurred = cv2.GaussianBlur(roi,
                                (CONFIG["BLUR_KERNEL"], CONFIG["BLUR_KERNEL"]), 0)
    edges   = cv2.Canny(blurred, CONFIG["CANNY_LOW"], CONFIG["CANNY_HIGH"])

    roi_height = CONFIG["ROI_Y_END"] - CONFIG["ROI_Y_START"]
    profile    = np.full(edges.shape[1], fill_value=float(roi_height), dtype=np.float32)

    for col in range(edges.shape[1]):
        rows = np.where(edges[:, col] > 0)[0]
        if len(rows):
            # CHANGE 1: take median of first few edge pixels instead of just rows[0]
            # This prevents a single noisy pixel from corrupting the reading
            profile[col] = np.median(rows[:5])

    profile = profile + CONFIG["ROI_Y_START"]

    # CHANGE 2: smooth the profile to remove noise spikes along the edge
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).flatten()

    return profile

# ──────────────────────────────────────────────
# CALIBRATION
# CHANGE 5: multi-image calibration — averages N images for a stable baseline
# ──────────────────────────────────────────────
def run_calibration(cam=None):
    n = CONFIG["CALIBRATION_IMAGES"]
    print(f"\n[CALIBRATE] You will capture {n} images of the BARE JIG (no PCB).")
    print(f"            These will be averaged into one stable baseline.")

    profiles = []

    for i in range(n):
        print(f"\n  Image {i+1}/{n} —", end=" ")
        rgb, gray = get_image(f"CALIBRATION {i+1}/{n}", cam)
        if gray is None:
            print(f"  [SKIPPED]")
            continue
        profile = get_edge_profile(gray)
        profiles.append(profile)
        print(f"  Profile captured. ({len(profiles)} so far)")

    if len(profiles) == 0:
        print("[ERROR] No calibration images captured. Calibration failed.")
        return None

    if len(profiles) < n:
        print(f"[WARN] Only {len(profiles)}/{n} images captured — baseline may be less stable.")

    # Average all profiles — this is much more stable than a single image
    baseline_profile = np.mean(profiles, axis=0)
    baseline_median  = float(np.median(baseline_profile))

    data = {
        "baseline_median_y": baseline_median,
        "baseline_per_col":  baseline_profile.tolist(),
        "timestamp":         datetime.now().isoformat(),
        "target_width":      CONFIG["TARGET_WIDTH"],
        "num_images":        len(profiles),
    }
    with open(CONFIG["BASELINE_FILE"], "w") as f:
        json.dump(data, f, indent=2)

    os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
    cal_img_path = os.path.join(CONFIG["LOG_DIR"], "calibration_image.jpg")
    if rgb is not None:
        cv2.imwrite(cal_img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print(f"\n[CALIBRATE] Done. Averaged {len(profiles)} images.")
    print(f"[CALIBRATE] Baseline median Y = {baseline_median:.1f} px")
    return baseline_profile

def load_baseline():
    if not os.path.exists(CONFIG["BASELINE_FILE"]):
        raise FileNotFoundError("No baseline.json — run Calibration first (option 2).")
    with open(CONFIG["BASELINE_FILE"]) as f:
        data = json.load(f)
    n = data.get("num_images", 1)
    print(f"[BASELINE] Loaded. Median Y = {data['baseline_median_y']:.1f} px  "
          f"(averaged from {n} image{'s' if n > 1 else ''})")
    return np.array(data["baseline_per_col"])

# ──────────────────────────────────────────────
# UPLIFT ANALYSIS
# CHANGE 3: vertical alignment correction before comparing
# CHANGE 4: smooth abs_diff before thresholding
# ──────────────────────────────────────────────
def analyze_uplift(current_profile, baseline_profile):
    if len(current_profile) != len(baseline_profile):
        print(f"[WARN] Profile length mismatch — resampling baseline.")
        baseline_profile = np.interp(
            np.linspace(0, 1, len(current_profile)),
            np.linspace(0, 1, len(baseline_profile)),
            baseline_profile
        )

    # CHANGE 3: vertical alignment correction
    # Removes any global vertical shift (e.g. camera moved slightly between shots)
    # so only LOCAL deviations (actual uplift) trigger detection
    offset          = np.median(current_profile - baseline_profile)
    current_profile = current_profile - offset

    # diff > 0 → board lifted above baseline
    # diff < 0 → board sunken below baseline
    # Both are errors
    diff     = baseline_profile - current_profile
    diff_mm  = diff / CONFIG["PX_PER_MM"]
    abs_diff = np.abs(diff)

    # CHANGE 4: smooth abs_diff to remove spike noise before thresholding
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

    region_fail    = len(fail_regions) > 0
    threshold_fail = max_abs_mm > CONFIG["MAX_MM_THRESHOLD"]
    passed         = not (region_fail or threshold_fail)

    return passed, fail_regions, diff, diff_mm, abs_diff, max_abs_mm, max_raw_mm

# ──────────────────────────────────────────────
# ANNOTATED OUTPUT IMAGE
# ──────────────────────────────────────────────
def save_annotated_image(color_img, current_profile, baseline_profile,
                          abs_diff, fail_regions, passed, timestamp):
    os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
    annotated = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    roi_x0    = CONFIG["ROI_X_START"]

    # Green = baseline edge
    for col_idx, by in enumerate(baseline_profile):
        x, y = col_idx + roi_x0, int(by)
        if 0 <= y < annotated.shape[0] and 0 <= x < annotated.shape[1]:
            annotated[y, x] = (0, 255, 0)

    # Current edge: yellow = ok, red = error
    for col_idx, cy in enumerate(current_profile):
        x, y  = col_idx + roi_x0, int(cy)
        color = (0, 0, 255) if abs_diff[col_idx] > CONFIG["UPLIFT_THRESHOLD_PX"] else (0, 255, 255)
        if 0 <= y < annotated.shape[0] and 0 <= x < annotated.shape[1]:
            annotated[y, x] = color

    # Orange ROI box
    cv2.rectangle(annotated,
                  (CONFIG["ROI_X_START"], CONFIG["ROI_Y_START"]),
                  (CONFIG["ROI_X_END"],   CONFIG["ROI_Y_END"]),
                  (0, 165, 255), 2)

    # Result label
    label = "PASS" if passed else f"FAIL — {len(fail_regions)} region(s)"
    cv2.putText(annotated, label, (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                (0, 200, 0) if passed else (0, 0, 255), 5)

    y_off = 160
    for r in fail_regions:
        direction = "↑ lifted" if r["direction"] == "lifted" else "↓ sunken"
        cv2.putText(annotated,
                    f"  {direction}  {r['max_uplift_mm']:.2f}mm  "
                    f"@ {r['x_start_mm']:.0f}–{r['x_end_mm']:.0f}mm",
                    (50, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        y_off += 55

    fname = os.path.join(CONFIG["LOG_DIR"],
                         f"{timestamp}_{'PASS' if passed else 'FAIL'}.jpg")
    cv2.imwrite(fname, annotated)
    print(f"[LOG] Annotated image → {fname}")
    return fname

# ──────────────────────────────────────────────
# CORE INSPECTION
# ──────────────────────────────────────────────
def inspect_image(color_img, gray, baseline_profile, source_label=""):
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_profile = get_edge_profile(gray)

    passed, fail_regions, diff, diff_mm, abs_diff, max_abs_mm, max_raw_mm = analyze_uplift(
        current_profile, baseline_profile
    )

    src = f"[{source_label}]  " if source_label else ""
    print("\n" + "=" * 60)
    print(f"  {src}RESULT:  {'✅  PASS' if passed else '❌  FAIL'}")
    print(f"  Max deviation: {max_abs_mm:.3f} mm  "
          f"({'lifted' if max_raw_mm > 0 else 'sunken'})")
    print(f"  Threshold:     {CONFIG['MAX_MM_THRESHOLD']} mm")
    if not passed:
        if fail_regions:
            print(f"  Fail regions ({len(fail_regions)}):")
            for i, r in enumerate(fail_regions, 1):
                direction = "↑ lifted" if r["direction"] == "lifted" else "↓ sunken"
                print(f"    {i}. {direction}  {r['max_uplift_mm']:.2f}mm  "
                      f"@ {r['x_start_mm']:.0f}–{r['x_end_mm']:.0f}mm from left")
        else:
            print("  [MAX_MM_THRESHOLD breach — spike detected]")
    print("=" * 60)

    img_path = save_annotated_image(
        color_img, current_profile, baseline_profile,
        abs_diff, fail_regions, passed, timestamp
    )

    os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
    log_entry = {
        "timestamp":    timestamp,
        "source":       source_label,
        "result":       "PASS" if passed else "FAIL",
        "max_mm":       round(max_abs_mm, 4),
        "fail_regions": fail_regions,
        "image":        img_path,
    }
    with open(os.path.join(CONFIG["LOG_DIR"], "inspection_log.jsonl"), "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return passed

# ──────────────────────────────────────────────
# MENU ACTIONS
# ──────────────────────────────────────────────
def action_inspect(cam, baseline_profile):
    rgb, gray = get_image("INSPECT", cam)
    if gray is not None:
        inspect_image(rgb, gray, baseline_profile,
                      source_label="CAMERA" if (cam and CAMERA_AVAILABLE) else "FILE")

def action_test_file(baseline_profile):
    while True:
        print("\n[FILE PICKER] Opening dialog...")
        path = pick_image_file(title="Select PCB image to inspect")
        if not path:
            print("[CANCELLED]")
            break
        try:
            rgb, gray = load_image_from_file(path)
            inspect_image(rgb, gray, baseline_profile,
                          source_label=os.path.basename(path))
        except Exception as e:
            print(f"[ERROR] {e}")
        if input("\nTest another image? [y/n]: ").strip().lower() != "y":
            break

# ──────────────────────────────────────────────
# MAIN MENU
# ──────────────────────────────────────────────
def main():
    load_roi_config()

    print("\n╔══════════════════════════════════════════════╗")
    print("║       PCB Warpage Detection System           ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  1. Inspect via camera                       ║")
    print("║  2. Calibrate  (bare jig — no PCB)           ║")
    print("║  3. Test on image file  (file picker)        ║")
    print("║  4. Set ROI interactively                    ║")
    print("║  5. Exit                                     ║")
    print("╚══════════════════════════════════════════════╝")

    cam              = None
    baseline_profile = None

    while True:
        choice = input("\nSelect option [1–5]: ").strip()

        if choice == "1":
            if not CAMERA_AVAILABLE:
                print("[ERROR] Camera not available. Use option 3.")
                continue
            if cam is None:
                cam = init_camera()
                if cam is None:
                    continue
            if baseline_profile is None:
                try:
                    baseline_profile = load_baseline()
                except FileNotFoundError as e:
                    print(f"[ERROR] {e}")
                    continue
            action_inspect(cam, baseline_profile)

        elif choice == "2":
            if baseline_profile is not None:
                if input("[WARN] Overwrite existing calibration? [y/n]: ").strip().lower() != "y":
                    continue
            baseline_profile = run_calibration(cam)

        elif choice == "3":
            if baseline_profile is None:
                try:
                    baseline_profile = load_baseline()
                except FileNotFoundError as e:
                    print(f"[ERROR] {e}")
                    continue
            action_test_file(baseline_profile)

        elif choice == "4":
            run_roi_setup(cam)

        elif choice == "5":
            break
        else:
            print("  Invalid choice. Enter 1–5.")

    if cam:
        cam.stop()
    print("\n[EXIT] Stopped.")

if __name__ == "__main__":
    main()
