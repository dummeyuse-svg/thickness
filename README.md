Most likely reasons (based on your code)
1. Duplicate cv2.namedWindow / cv2.imshow calls

Inside your ROI function:

cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
...
while True:
    cv2.imshow("ROI Selector", draw_overlay())

👉 If while typing manually you:

accidentally duplicated cv2.imshow(...)
or used a slightly different window name somewhere (like "ROI selector" vs "ROI Selector")

➡️ OpenCV treats them as two separate windows

⚠️ One of them might receive:

no image (→ black window)
or an uninitialized frame
2. draw_overlay() returning empty/incorrect frame

If you mistyped something here:

out = display_base.copy()

If display_base became:

None
or wrong variable name

➡️ Then:

one window shows proper frame
another shows empty/black buffer
3. Two loops / function called twice

If you accidentally:

called select_roi_interactively() twice
or had a duplicate block of code

➡️ Two windows open simultaneously

4. Typo in window name (VERY common)

Even this tiny difference creates two windows:

cv2.namedWindow("ROI Selector", ...)
cv2.imshow("ROI_Selector", ...)  # underscore!

➡️ Result:

One window = real image
One window = blank
5. Missing cv2.destroyAllWindows() (less likely here)

You do have:

cv2.destroyAllWindows()

But if you mistyped it earlier:

old window stays
new one opens


🔁 Replace this inside get_edge_profile():
for col in range(edges.shape[1]):
    rows = np.where(edges[:, col] > 0)[0]
    if len(rows):
        profile[col] = rows[0]
✅ With this:
for col in range(edges.shape[1]):
    rows = np.where(edges[:, col] > 0)[0]
    if len(rows):
        # Take median of first few edge pixels (robust against noise)
        profile[col] = np.median(rows[:5])

Replace your return line:
return profile + CONFIG["ROI_Y_START"]
✅ With:
profile = profile + CONFIG["ROI_Y_START"]

# Smooth the profile to remove noise spikes
profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).flatten()

return profile


nside analyze_uplift()
🔁 BEFORE this line:
diff = baseline_profile - current_profile
✅ ADD THIS:
# ALIGNMENT FIX (very important)
offset = np.median(current_profile - baseline_profile)
current_profile = current_profile - offset

abs_diff = np.abs(diff)
✅ With:
abs_diff = np.abs(diff)

# Smooth diff to remove spike noise
abs_diff = cv2.GaussianBlur(abs_diff.reshape(-1, 1), (1, 11), 0).flatten()


optional but highly recommened
"UPLIFT_THRESHOLD_PX": 6,

def run_calibration(cam=None):
    print("\n[CALIBRATE] Capture 5–10 images of BARE JIG")

    profiles = []

    for i in range(5):
        print(f"\nCapture calibration image {i+1}/5")
        rgb, gray = get_image(f"CALIBRATION {i+1}", cam)
        if gray is None:
            continue

        profile = get_edge_profile(gray)
        profiles.append(profile)

    if len(profiles) == 0:
        print("[ERROR] No calibration images captured")
        return None

    baseline_profile = np.mean(profiles, axis=0)
    baseline_median = float(np.median(baseline_profile))

    data = {
        "baseline_median_y": baseline_median,
        "baseline_per_col":  baseline_profile.tolist(),
        "timestamp":         datetime.now().isoformat(),
        "target_width":      CONFIG["TARGET_WIDTH"],
    }

    with open(CONFIG["BASELINE_FILE"], "w") as f:
        json.dump(data, f, indent=2)

    print(f"[CALIBRATE] Done using {len(profiles)} images")
    return baseline_profile

