import os
import pytesseract
import PIL.Image
from pytesseract import Output
import cv2
import pandas as pd

# Optional: set explicit tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

myconfig = r"--psm 11 --oem 3"

img_path = "/home/kirat/coding/pecathon/image copy 6.png"
if not os.path.isfile(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

img = cv2.imread(img_path)
if img is None:
    raise RuntimeError("cv2.imread returned None. Check path or permissions.")

data = pytesseract.image_to_data(img, config=myconfig, output_type=Output.DICT)

rows = []
for i in range(len(data['text'])):
    text = data['text'][i].strip()
    try:
        conf = float(data['conf'][i])
    except ValueError:
        conf = -1
    if text and conf >= 50:  # lowered threshold
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        rows.append({
            "text": text,
            "confidence": conf,
            "left": x,
            "top": y,
            "width": w,
            "height": h
        })
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

out_csv = "output/ocr_results.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"Saved {len(rows)} rows to {out_csv}")

# Save annotated image (avoid imshow in headless)
annotated_path = img_path.split("/")[-1]
cv2.imwrite(annotated_path, img)
print(f"Annotated image saved to {annotated_path}")
