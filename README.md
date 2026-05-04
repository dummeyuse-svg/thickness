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



pip show PyQt5 2>/dev/null | head -3; python3 -c "import PyQt5" 2>&1 | head -3





steps to do offline setup (5-4-2026)
🥇 Step 1: Install Python (if not already)

On your work laptop:

python --version
If it works → ✅ skip
If not → install Python (offline installer if needed)
🥈 Step 2: Extract your project

Unzip your folder somewhere like:

Desktop/mttr_project/

Inside it should have:

app.py
clean_excel.py
index.html
offline-packages/
chroma_db/ (if already created)
requirements.txt (if you made it)
🥉 Step 3: Create virtual environment (VERY IMPORTANT)

Inside that folder:

python -m venv venv

Activate it:

venv\Scripts\activate
🏗️ Step 4: Install dependencies (OFFLINE)

Assuming your folder is offline-packages/:

pip install --no-index --find-links=offline-packages -r requirements.txt

👉 If no requirements.txt:

pip install --no-index --find-links=offline-packages *
🤖 Step 5: Install Ollama

Run your Ollama setup file:

Install normally
Finish installation
🧠 Step 6: Add your model (IMPORTANT)

You said you already have the model — good.

Option A (BEST – if your model folder exists)

Copy this from your personal laptop:

C:\Users\<your-user>\.ollama

Paste into work laptop:

C:\Users\<work-user>\.ollama
Then verify:
ollama list

👉 You should see:

phi3
or
phi3:mini
⚡ Step 7: Start Ollama
ollama run phi3

👉 Keep this terminal open

🧪 Step 8: Run your backend

In another terminal:

uvicorn app:app --host 127.0.0.1 --port 8000
🌐 Step 9: Open frontend

Open:

index.html

OR go to:

http://127.0.0.1:8000/docs
🚨 Common failure points (read this carefully)
❌ Problem 1: ModuleNotFoundError

👉 Fix:

pip install --no-index --find-links=offline-packages <missing-package>
❌ Problem 2: Ollama timeout (you already saw this)

👉 Fix:

Make sure:
ollama run phi3

is running

❌ Problem 3: model not found

👉 Fix:

ollama list

If empty → model copy failed

❌ Problem 4: ChromaDB not working

👉 Ensure:

chroma_db/

folder is in same directory

