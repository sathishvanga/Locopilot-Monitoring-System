"""Visualize test labels on 10 sample frames."""
import cv2, os
from pathlib import Path

FRAMES_DIR = "/Users/satishvanga/Documents/frames"
LABELS_DIR = "/Users/satishvanga/Documents/training_cvvrs/test_labels"
OUTPUT_DIR = "/Users/satishvanga/Documents/training_cvvrs/test_review"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["person", "cell_phone", "book", "cup", "bottle",
               "backpack", "handbag", "suitcase", "radio_handset"]
COLORS = [(0,255,0),(0,0,255),(255,0,0),(0,255,255),(255,165,0),
          (128,0,128),(255,192,203),(0,128,128),(255,255,0)]

for label_path in sorted(Path(LABELS_DIR).glob("*.txt")):
    frame_path = os.path.join(FRAMES_DIR, label_path.stem + ".jpg")
    if not os.path.exists(frame_path):
        continue
    img = cv2.imread(frame_path)
    h, w = img.shape[:2]
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: continue
            cid = int(parts[0])
            cx,cy,bw,bh = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
            x1,y1 = int((cx-bw/2)*w), int((cy-bh/2)*h)
            x2,y2 = int((cx+bw/2)*w), int((cy+bh/2)*h)
            color = COLORS[cid] if cid < len(COLORS) else (255,255,255)
            label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls_{cid}"
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,1)
            cv2.rectangle(img,(x1,y1-th-6),(x1+tw,y1),color,-1)
            cv2.putText(img,label,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, label_path.stem + ".jpg"), img)
    print(f"Saved: {label_path.stem}.jpg")
