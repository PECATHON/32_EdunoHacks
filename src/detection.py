from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import cv2,os
from PIL import Image
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import pandas as pd
import json, os
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
def table_extraction(image_path):
        # img = cv2.imread(img_path)

        # image_path = image_path
        output_dir = image_path.split('/')[-1]
        os.makedirs(output_dir, exist_ok=True)

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path} or cv2 failed to read it.")

        # Resize if very large for more stable OCR
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > 1200:
            scale = 1200.0 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        # run pytesseract word-level OCR
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)

        # collect valid boxes (filter low confidence)
        boxes = []
        for i, txt in enumerate(data['text']):
            t = txt.strip()
            try:
                conf = float(data['conf'][i])
            except:
                conf = -1.0
            if t and conf >= 30:  # accept moderately confident words
                left = int(data['left'][i]); top = int(data['top'][i])
                width = int(data['width'][i]); height = int(data['height'][i])
                cx = left + width//2
                cy = top + height//2
                boxes.append({'text': t, 'conf': conf, 'left': left, 'top': top,
                            'width': width, 'height': height, 'cx': cx, 'cy': cy})

        if not boxes:
            raise RuntimeError("No OCR boxes found with sufficient confidence. Try lowering threshold or improving image quality.")

        # cluster rows by cy (vertical clustering)
        boxes_sorted = sorted(boxes, key=lambda b: b['cy'])
        row_tol = max(8, int(h * 0.02))
        rows = []
        current_row = [boxes_sorted[0]]
        for b in boxes_sorted[1:]:
            if abs(b['cy'] - current_row[-1]['cy']) <= row_tol:
                current_row.append(b)
            else:
                rows.append(current_row)
                current_row = [b]
        rows.append(current_row)

        # For columns, simple KMeans on cx to find likely column centers.
        # We'll try k from 2..8 and pick the k that makes sense (silhouette-like by within-cluster variance)
        def kmeans_1d(xs, k, iters=30):
            xs = np.array(xs, dtype=float)
            # init centroids as quantiles
            cents = np.percentile(xs, np.linspace(0,100,k))
            for _ in range(iters):
                # assign
                idx = np.argmin(np.abs(xs[:,None] - cents[None,:]), axis=1)
                new_cents = np.array([xs[idx==j].mean() if np.any(idx==j) else cents[j] for j in range(k)])
                if np.allclose(new_cents, cents, atol=1e-2, equal_nan=True):
                    break
                cents = new_cents
            return cents, idx

        all_cx = [b['cx'] for b in boxes]
        best_k = None
        best_score = None
        best_centroids = None
        best_assign = None
        for k in range(2,7):
            cents, assign = kmeans_1d(all_cx, k)
            # compute within-cluster sum of distances
            wcss = sum(abs(all_cx[i] - cents[assign[i]]) for i in range(len(all_cx)))
            # penalize tiny clusters
            counts = np.bincount(assign, minlength=k)
            penalty = sum(1 for c in counts if c < 2) * 1000
            score = wcss + penalty
            if best_score is None or score < best_score:
                best_score = score
                best_k = k
                best_centroids = cents
                best_assign = assign

        k = best_k
        centroids = best_centroids
        # map each box to a column index based on centroid closest
        for i, b in enumerate(boxes):
            # find nearest centroid
            col_idx = int(np.argmin([abs(b['cx'] - c) for c in centroids]))
            b['col'] = col_idx

        # Build table grid with number of rows = len(rows), cols = k
        table = []
        for r in rows:
            # sort words left->right
            row_cells = [""] * k
            # for each box in this row, find its col and append text
            for b in sorted(r, key=lambda x: x['left']):
                col = b.get('col', 0)
                if row_cells[col]:
                    row_cells[col] += " " + b['text']
                else:
                    row_cells[col] = b['text']
            table.append(row_cells)

        # Post-process: remove rows that are mostly empty, strip spaces
        clean_rows = []
        for row in table:
            stripped = [c.strip() for c in row]
            # drop row if all empty
            if any(c for c in stripped):
                clean_rows.append(stripped)

        # Try to identify header row as the first row which contains text like 'Type' or 'Graph Terminology' etc.
        # If the first row is a title with single long cell, skip it and use next row as header.
        if clean_rows and len(clean_rows[0])==1 and len(clean_rows)>1 and len(clean_rows[1])>1:
            # possibly title row, drop it
            clean_rows = clean_rows[1:]

        # Normalize column count by padding
        max_cols = max(len(r) for r in clean_rows)
        norm_rows = [r + [""]*(max_cols - len(r)) for r in clean_rows]

        # Save CSV and JSON
        page = 1
        table_index = 0
        csv_path = os.path.join(output_dir, f"page_{page}_table_{table_index}.csv")
        json_path = os.path.join(output_dir, f"page_{page}_table_{table_index}.json")

        df = pd.DataFrame(norm_rows)
        df.to_csv(csv_path, index=False, header=False)

        json_obj = {"page": page, "table_index": table_index, "bbox": [0,0,w,h], "rows": norm_rows}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, ensure_ascii=False, indent=2)

        print("Saved CSV:", csv_path)
        print("Saved JSON:", json_path)
        # show a preview DataFrame to the user using ace_tools display if available
        try:
            import ace_tools as tools
            tools.display_dataframe_to_user("Extracted Table Preview", df)
        except Exception:
            # fallback: print first few rows
            print(df.head(20).to_string())

        # also save an annotated image showing detected boxes for debugging
        annot = img.copy()
        for b in boxes:
            l,t = b['left'], b['top']
            rr = b['height']; cc = b['width']
            cv2.rectangle(annot, (l,t), (l+cc, t+rr), (0,255,0), 1)
            # put small label of col index
            col = b.get('col', 0)
            cv2.putText(annot, str(col), (b['left'], b['top']-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)

        annot_path = os.path.join(output_dir, "annotated.png")
        cv2.imwrite(annot_path, annot)
        print("Annotated image saved to:", annot_path)

        # list files created
        os.listdir(output_dir)[:50]
def textbox_extraction(img_path):

    myconfig = r"--psm 11 --oem 3"

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
# Load the config file
cfg = get_cfg()
cfg.merge_from_file("configs/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "models/mask_rcnn_R_50_FPN_3x.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

# Set up metadata for PubLayNet classes
MetadataCatalog.get("publaynet_val").thing_classes = ["text", "title", "list", "table", "figure"]

predictor = DefaultPredictor(cfg)
im = cv2.imread("input/table.jpg")
outputs = predictor(im)

# # Visualize the results
# from detectron2.utils.visualizer import Visualizer
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("publaynet_val"), scale=1.2)
# v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow("Result", v.get_image()[:, :, ::-1])
# cv2.waitKey(0)
# Extract instances
instances = outputs["instances"].to("cpu")
boxes = instances.pred_boxes.tensor.numpy()
classes = instances.pred_classes.numpy()
class_names = MetadataCatalog.get("publaynet_val").thing_classes

# Create output folder

# Crop and save each region
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    crop = im[y1:y2, x1:x2]
    label = class_names[classes[i]]
    os.makedirs(f"output/{label}", exist_ok=True)
    filename = f"output/{label}/{i:03d}.png"
    cv2.imwrite(filename, crop)
    if label=="table":
        table_extraction(filename)
    elif label=="text":
        textbox_extraction(filename)
