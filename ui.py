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
