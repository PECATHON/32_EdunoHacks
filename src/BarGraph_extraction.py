import os
import cv2
import numpy as np
import pandas as pd

# CONFIG PATHS


BAR_CHART_DIR = r"C:\Users\lenovo\OneDrive\Desktop\classify\data\output_folder\bar_charts"
CSV_OUTPUT_DIR = r"C:\Users\lenovo\OneDrive\Desktop\classify\data\output_folder\bar_charts\csv"

os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)


# SINGLE BAR CHART OUTLINE DETECTION (works for any color)
def detect_single_bars(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image: " + image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Stronger edge detection for hollow bars
    edges = cv2.Canny(gray, 10, 60)

    # Dilate to strengthen thin outlines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find all contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bars = []
    H, W = img.shape[:2]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        # Filter tiny noise
        if h < H * 0.10:
            continue

        # Bars are tall, not wide
        if h > w * 2:
            bars.append((x, y, w, h))

    # Sort left→right
    bars = sorted(bars, key=lambda b: b[0])

    # Remove duplicate outlines (left & right edges)
    cleaned = []
    for (x,y,w,h) in bars:
        if not cleaned:
            cleaned.append((x,y,w,h))
        else:
            last_x = cleaned[-1][0]
            if abs(x - last_x) > 10:  # prevents double counting
                cleaned.append((x,y,w,h))

    # Build DataFrame
    df = pd.DataFrame([
        {"Category": f"Bar {i+1}", "PixelHeight": h}
        for i, (x,y,w,h) in enumerate(cleaned)
    ])

    return df

# PROCESS ALL IMAGES + SAVE CSVs
def process_all_single_bar_charts():
    processed = 0

    for fname in os.listdir(BAR_CHART_DIR):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")) and fname.lower() != "csv":
            img_path = os.path.join(BAR_CHART_DIR, fname)

            print(f"Processing {fname}...")

            try:
                df = detect_single_bars(img_path)

                out_name = os.path.splitext(fname)[0] + ".csv"
                out_csv_path = os.path.join(CSV_OUTPUT_DIR, out_name)

                df.to_csv(out_csv_path, index=False)

                print(f"  ✔ Saved CSV: {out_csv_path}")
                processed += 1

            except Exception as e:
                print(f" Error processing {fname}: {e}")

    print(f"\nDone! Processed {processed} images.")
    print(f"CSV files saved in: {CSV_OUTPUT_DIR}")

# MAIN

if __name__ == "__main__":
    process_all_single_bar_charts()
