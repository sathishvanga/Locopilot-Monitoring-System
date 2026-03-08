"""
Auto-label locomotive cabin CCTV frames using Grounding DINO on GPU.
Text-prompted zero-shot object detection -> YOLO format labels.
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# --- Config ---
FRAMES_DIR = "/home/admin1/auto_label/frames"
OUTPUT_BASE = "/home/admin1/auto_label/training_cvvrs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class definitions with custom prompts for Grounding DINO
CLASS_PROMPTS = {
    0: ("person", "person sitting in train locomotive cabin"),
    1: ("cell_phone", "small rectangular mobile phone held in hand"),
    2: ("book", "open log book or register on desk surface"),
    3: ("cup", "drinking cup or mug held in hand or on surface"),
    4: ("bottle", "steel thermos flask or water bottle on desk"),
    5: ("backpack", "backpack or rucksack bag on floor or seat"),
    6: ("handbag", "small handbag or carry bag"),
    7: ("suitcase", "suitcase or luggage bag on floor"),
    8: ("radio_handset", "handheld radio transceiver or walkie talkie held near face"),
}

# Confidence thresholds per class
CLASS_THRESHOLDS = {
    "person": 0.35,
    "cell_phone": 0.40,
    "book": 0.30,
    "cup": 0.40,
    "bottle": 0.28,
    "backpack": 0.35,
    "handbag": 0.30,
    "suitcase": 0.35,
    "radio_handset": 0.55,
}

# Train/val split
TRAIN_RATIO = 0.85


def setup_directories():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_BASE, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE, split, "labels"), exist_ok=True)


def load_model():
    model_id = "IDEA-Research/grounding-dino-base"
    print(f"Loading Grounding DINO: {model_id} on {DEVICE}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id, torch_dtype=torch.float32
    ).to(DEVICE)
    model.eval()
    print("Model loaded.")
    return processor, model


def detect_all_classes(image_pil, processor, model):
    """Run detection for all classes in a single pass using concatenated prompt."""
    # Build combined text prompt
    all_prompts = []
    prompt_to_class = {}
    for class_id, (class_name, prompt) in CLASS_PROMPTS.items():
        all_prompts.append(prompt)
        prompt_to_class[prompt.lower()] = (class_id, class_name)

    combined_text = " . ".join(all_prompts) + " ."

    inputs = processor(images=image_pil, text=combined_text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    h, w = image_pil.size[::-1]  # PIL size is (w, h)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=0.25,
        text_threshold=0.25,
        target_sizes=[(h, w)],
    )[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        label_lower = label.strip().lower()

        # Match label to class
        matched_class_id = None
        matched_class_name = None
        for prompt, (cid, cname) in prompt_to_class.items():
            if label_lower in prompt or prompt.startswith(label_lower):
                matched_class_id = cid
                matched_class_name = cname
                break

        # Fallback: keyword matching
        if matched_class_id is None:
            keyword_map = {
                "person": (0, "person"),
                "phone": (1, "cell_phone"),
                "mobile": (1, "cell_phone"),
                "book": (2, "book"),
                "log": (2, "book"),
                "register": (2, "book"),
                "cup": (3, "cup"),
                "mug": (3, "cup"),
                "bottle": (4, "bottle"),
                "thermos": (4, "bottle"),
                "flask": (4, "bottle"),
                "backpack": (5, "backpack"),
                "rucksack": (5, "backpack"),
                "handbag": (6, "handbag"),
                "bag": (6, "handbag"),
                "suitcase": (7, "suitcase"),
                "luggage": (7, "suitcase"),
                "radio": (8, "radio_handset"),
                "walkie": (8, "radio_handset"),
                "transceiver": (8, "radio_handset"),
            }
            for keyword, (cid, cname) in keyword_map.items():
                if keyword in label_lower:
                    matched_class_id = cid
                    matched_class_name = cname
                    break

        if matched_class_id is None:
            continue

        # Apply per-class threshold
        threshold = CLASS_THRESHOLDS.get(matched_class_name, 0.25)
        if float(score) < threshold:
            continue

        x1, y1, x2, y2 = box.cpu().float().numpy()
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        # Size filters to reject FPs (normalized coords, 1280x720 frame)
        # Small objects like cell_phone/cup must be reasonable size, not tiny knobs
        MIN_SIZE = {
            "cell_phone": (0.02, 0.03),  # min ~26x22 px
            "cup": (0.02, 0.03),
            "bottle": (0.015, 0.04),
            "radio_handset": (0.03, 0.04),
            "book": (0.05, 0.05),
        }
        MAX_SIZE = {
            "cell_phone": (0.15, 0.20),   # max ~192x144 px — reject huge FPs
            "cup": (0.15, 0.20),
            "radio_handset": (0.25, 0.30),
            "book": (0.30, 0.45),           # real log book ~250x250px max
            "person": (0.85, 0.95),         # person shouldn't be entire frame
        }
        if matched_class_name in MIN_SIZE:
            min_w, min_h = MIN_SIZE[matched_class_name]
            if bw < min_w or bh < min_h:
                continue
        if matched_class_name in MAX_SIZE:
            max_w, max_h = MAX_SIZE[matched_class_name]
            if bw > max_w or bh > max_h:
                continue

        # Aspect ratio filter: real bottles are tall+narrow (h/w>4), FP canisters are squatter
        if matched_class_name == "bottle" and bw > 0 and (bh / bw) < 4.0:
            continue

        # Zone suppression for known static FP regions in locomotive cabin
        suppress = False

        # FP1: cell_phone near IPCamera watermark (bottom-right dashboard equipment)
        if matched_class_name == "cell_phone" and cx > 0.78 and cy > 0.78:
            suppress = True

        # FP2: handbag/backpack at right edge (locomotive fixtures)
        if matched_class_name in ("handbag", "backpack") and cx > 0.92:
            suppress = True

        # FP3: bottle at unusual positions (not the thermos at ~0.70, 0.37)
        if matched_class_name == "bottle" and (cy > 0.55 or cx > 0.85):
            suppress = True

        # FP4: suitcase in central area (thermos/equipment confusion)
        if matched_class_name == "suitcase" and cx < 0.80:
            suppress = True

        if suppress:
            continue

        if 0.005 < bw < 1.0 and 0.005 < bh < 1.0:
            detections.append((matched_class_id, cx, cy, bw, bh, float(score)))

    return detections


def main():
    print("=" * 60)
    print("Grounding DINO Auto-Labeling (GPU)")
    print("=" * 60)

    setup_directories()
    processor, model = load_model()

    frame_files = sorted(Path(FRAMES_DIR).glob("*.jpg"))
    total = len(frame_files)
    split_idx = int(total * TRAIN_RATIO)
    train_set = set(f.name for f in frame_files[:split_idx])
    print(f"Frames: {total} (train: {split_idx}, val: {total - split_idx})")

    stats = {name: 0 for _, (name, _) in CLASS_PROMPTS.items()}
    labeled_count = 0

    for frame_file in tqdm(frame_files, desc="Auto-labeling"):
        split = "train" if frame_file.name in train_set else "val"

        image_pil = Image.open(frame_file).convert("RGB")
        detections = detect_all_classes(image_pil, processor, model)

        # Symlink image to split dir
        img_dst = os.path.join(OUTPUT_BASE, split, "images", frame_file.name)
        if not os.path.exists(img_dst):
            try:
                os.symlink(str(frame_file), img_dst)
            except OSError:
                import shutil
                shutil.copy2(str(frame_file), img_dst)

        # Write YOLO label
        label_path = os.path.join(OUTPUT_BASE, split, "labels", f"{frame_file.stem}.txt")
        with open(label_path, "w") as f:
            for class_id, cx, cy, bw, bh, score in detections:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                class_name = CLASS_PROMPTS[class_id][0]
                stats[class_name] += 1

        if detections:
            labeled_count += 1

    # Write dataset.yaml
    class_names = [name for _, (name, _) in sorted(CLASS_PROMPTS.items())]
    yaml_path = os.path.join(OUTPUT_BASE, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_BASE}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    # Write classes.txt
    with open(os.path.join(OUTPUT_BASE, "classes.txt"), "w") as f:
        for name in class_names:
            f.write(f"{name}\n")

    print("\n" + "=" * 60)
    print("Detection Statistics:")
    print("-" * 40)
    for name, count in stats.items():
        print(f"  {name:20s}: {count:5d}")
    print(f"  {'TOTAL':20s}: {sum(stats.values()):5d}")
    print(f"\n  Frames with detections: {labeled_count}/{total}")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_BASE}")
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
