"""
Quick test: Run RTMPose-M on a sample CCTV frame and save annotated output.
Uses rtmlib for lightweight inference (numpy + opencv + onnxruntime only).
"""

import cv2
import numpy as np
from rtmlib import Body, draw_skeleton

# --- Config ---
INPUT_IMAGE = "object_detections_review/frame_00026550_backpack.jpg"
OUTPUT_DIR = "rtm debug frames"
OUTPUT_IMAGE = f"{OUTPUT_DIR}/frame_00026550_rtmpose.jpg"

# RTMPose-M with YOLOX-m detector (auto-downloads ONNX weights)
# mode='balanced' = YOLOX-m detector + RTMPose-M pose (256x192)
# to_openpose=False gives COCO-17 keypoints (same as YOLO-Pose)
body = Body(
    mode='balanced',          # YOLOX-m + RTMPose-M
    to_openpose=False,        # COCO-17 format (matches YOLO-Pose keypoints)
    backend='onnxruntime',
    device='cpu',
)

# COCO-17 keypoint names
KEYPOINT_NAMES = [
    'Nose', 'L Eye', 'R Eye', 'L Ear', 'R Ear',
    'L Shoulder', 'R Shoulder', 'L Elbow', 'R Elbow',
    'L Wrist', 'R Wrist', 'L Hip', 'R Hip',
    'L Knee', 'R Knee', 'L Ankle', 'R Ankle'
]

# Colors for keypoints
KP_COLOR = (0, 255, 0)       # Green dots
LABEL_COLOR = (255, 255, 0)  # Cyan labels
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6),                                      # Shoulders
    (5, 7), (7, 9),                              # Left arm
    (6, 8), (8, 10),                             # Right arm
    (5, 11), (6, 12),                            # Torso
    (11, 12),                                    # Hips
    (11, 13), (13, 15),                          # Left leg
    (12, 14), (14, 16),                          # Right leg
]

def annotate_frame(img, keypoints, scores):
    """Draw keypoints, skeleton, and labels on image."""
    annotated = img.copy()

    for person_idx, (kps, scrs) in enumerate(zip(keypoints, scores)):
        # Draw skeleton lines first (behind keypoints)
        for (i, j) in SKELETON_PAIRS:
            if i < len(kps) and j < len(kps):
                conf_i = scrs[i] if i < len(scrs) else 0
                conf_j = scrs[j] if j < len(scrs) else 0
                if conf_i > 0.3 and conf_j > 0.3:
                    pt1 = (int(kps[i][0]), int(kps[i][1]))
                    pt2 = (int(kps[j][0]), int(kps[j][1]))
                    cv2.line(annotated, pt1, pt2, (0, 255, 255), 2)

        # Draw keypoints and labels
        for kp_idx, (kp, score) in enumerate(zip(kps, scrs)):
            x, y = int(kp[0]), int(kp[1])
            conf = float(score)

            if conf > 0.3:  # Only draw confident keypoints
                # Draw keypoint circle
                cv2.circle(annotated, (x, y), 5, KP_COLOR, -1)
                cv2.circle(annotated, (x, y), 5, (0, 0, 0), 1)

                # Draw label with confidence
                label = f"{KEYPOINT_NAMES[kp_idx]} {conf:.2f}"
                # Offset label to avoid overlap
                label_x = x + 8
                label_y = y - 5

                # Background for readability
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated, (label_x - 1, label_y - th - 2),
                            (label_x + tw + 1, label_y + 2), (0, 0, 0), -1)
                cv2.putText(annotated, label, (label_x, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, LABEL_COLOR, 1)

        # Person label
        valid_kps = [(kps[i][0], kps[i][1]) for i in range(len(kps)) if scrs[i] > 0.3]
        if valid_kps:
            min_y = min(p[1] for p in valid_kps)
            center_x = np.mean([p[0] for p in valid_kps])
            cv2.putText(annotated, f"Person {person_idx}",
                       (int(center_x) - 30, int(min_y) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    return annotated


def main():
    # Load image
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"ERROR: Could not load {INPUT_IMAGE}")
        return

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    # Run RTMPose
    print("Running RTMPose-M inference...")
    keypoints, scores = body(img)

    print(f"\nDetected {len(keypoints)} person(s)")

    # Print detailed keypoint info
    for person_idx, (kps, scrs) in enumerate(zip(keypoints, scores)):
        print(f"\n--- Person {person_idx} ---")
        for kp_idx, (kp, score) in enumerate(zip(kps, scrs)):
            conf = float(score)
            status = "OK" if conf > 0.3 else "LOW" if conf > 0.1 else "SKIP"
            print(f"  {KEYPOINT_NAMES[kp_idx]:>12s}: ({kp[0]:7.1f}, {kp[1]:7.1f})  conf={conf:.3f}  [{status}]")

    # Draw annotated frame
    annotated = annotate_frame(img, keypoints, scores)

    # Also save the rtmlib default visualization for comparison
    rtmlib_viz = draw_skeleton(img.copy(), keypoints, scores, kpt_thr=0.3)

    # Save outputs
    cv2.imwrite(OUTPUT_IMAGE, annotated)
    cv2.imwrite(f"{OUTPUT_DIR}/frame_00026550_rtmpose_rtmlib_default.jpg", rtmlib_viz)

    print(f"\nSaved: {OUTPUT_IMAGE}")
    print(f"Saved: {OUTPUT_DIR}/frame_00026550_rtmpose_rtmlib_default.jpg")


if __name__ == "__main__":
    main()
