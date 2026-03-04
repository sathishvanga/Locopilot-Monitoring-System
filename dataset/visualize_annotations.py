"""Visualize YOLO annotations overlaid on images to verify correctness."""
import cv2
import os
import glob

CLASSES = {0: "person", 1: "cell_phone", 2: "book", 3: "cup", 4: "bottle"}
COLORS = {
    0: (0, 255, 0),    # person - green
    1: (0, 0, 255),    # cell_phone - red
    2: (255, 0, 0),    # book - blue
    3: (0, 255, 255),  # cup - yellow
    4: (255, 0, 255),  # bottle - magenta
}

def draw_annotations(img_path, lbl_path, out_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert normalized to pixel coords
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            color = COLORS.get(cls_id, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = CLASSES.get(cls_id, str(cls_id))
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    base = "/Users/satishvanga/Desktop/Practice/dataset"
    out_dir = os.path.join(base, "visualized")
    os.makedirs(out_dir, exist_ok=True)

    for split in ["train", "val"]:
        for img_path in sorted(glob.glob(os.path.join(base, split, "images", "*.jpg"))):
            name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(base, split, "labels", f"{name}.txt")
            if os.path.exists(lbl_path):
                out_path = os.path.join(out_dir, f"{split}_{name}.jpg")
                draw_annotations(img_path, lbl_path, out_path)
