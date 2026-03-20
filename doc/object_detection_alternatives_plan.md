# Object Detection Alternatives Plan

## Date: 2026-03-18

## Problem Statement

YOLO26n (COCO-pretrained) fails to detect **book**, **cell phone**, and **cup/bottle** in overhead CCTV frames from train locomotive cabins.

### Root Cause: Domain Gap

| Factor | Detail |
|--------|--------|
| Camera angle | Overhead ~60-degree (COCO = eye-level) |
| Object size | 30-100px in 1280x720 frame |
| Environment | Industrial clutter (buttons, knobs, switches) |
| Lighting | Low light + backlighting from door |
| Compression | H.264 artifacts blur small object edges |

### Current YOLO26n Detection Results (conf >= 0.15)

| Object | Expected | Detected As | Status |
|--------|----------|-------------|--------|
| Person | Yes | `person` (0.63-0.90) | OK |
| Book/logbook | Yes | Not detected | MISSED |
| Cell phone | Yes | Not detected | MISSED |
| Cup/bottle | Yes | Not detected | MISSED |
| Bag | Yes | `suitcase` (0.23-0.87) | Partial (wrong class) |
| Chair | Yes | `chair` (0.36-0.45) | OK |
| Monitor/screen | Yes | `tv` (0.22-0.67) | OK |

---

## Alternative Models (Ranked by Implementation Effort)

### Option 1: SAHI + YOLO26l (Quick Win - No Retraining)

- **What**: Sliced Aided Hyper Inference — divide frame into overlapping crops, run YOLO on each, merge results
- **Why**: Small objects become larger relative to each crop; YOLO26l (large) has better small-object recall than nano
- **Install**: `pip install sahi` (partial config already exists in codebase)
- **Expected gain**: +5-15% recall on small objects
- **Speed**: ~5-10 FPS
- **Effort**: Low (1-2 days)
- **HuggingFace task**: `object-detection`
- **Reference**: https://github.com/obss/sahi

### Option 2: Grounding DINO (Zero-Shot, Text-Prompted - No Retraining)

- **What**: Open-vocabulary detector — prompt with `"book . cell phone . bag . cup . bottle"` and it finds them
- **Why**: Best zero-shot accuracy; handles unusual angles better than COCO-only models due to language grounding
- **Install**: `pip install groundingdino` or via HuggingFace transformers
- **Models**:
  - `IDEA-Research/grounding-dino-tiny` — faster, lighter
  - `IDEA-Research/grounding-dino-base` — higher accuracy
- **Expected gain**: Significant improvement on book/phone/cup detection
- **Speed**: ~2-5 FPS (best for voting verification stage, not every-frame)
- **Effort**: Medium (2-3 days)
- **HuggingFace task**: `zero-shot-object-detection`
- **HuggingFace**: https://huggingface.co/IDEA-Research/grounding-dino-base

### Option 3: Florence-2 (Zero-Shot VLM - No Retraining)

- **What**: Microsoft's lightweight vision-language model with object detection capability
- **Why**: Smaller and faster than Grounding DINO; strong zero-shot; supports multiple vision tasks
- **Install**: `pip install transformers` (already available)
- **Models**:
  - `microsoft/Florence-2-base` — 0.23B params, faster
  - `microsoft/Florence-2-large` — 0.77B params, more accurate
- **Expected gain**: Good improvement, especially on common objects from unusual angles
- **Speed**: ~3-5 FPS
- **Effort**: Medium (2-3 days)
- **HuggingFace task**: `image-text-to-text` (multi-task VLM)
- **HuggingFace**: https://huggingface.co/microsoft/Florence-2-large

### Option 4: OWLv2 (Zero-Shot - No Retraining)

- **What**: Google's open-vocabulary detector with self-training approach
- **Why**: Good generalization to unusual viewpoints; efficient scaling
- **Install**: `pip install transformers` (already available)
- **Models**:
  - `google/owlv2-base-patch16-ensemble` — balanced
  - `google/owlv2-large-patch14-ensemble` — higher accuracy
- **Expected gain**: Moderate improvement
- **Speed**: ~3-5 FPS
- **Effort**: Medium (2-3 days)
- **HuggingFace task**: `zero-shot-object-detection`
- **HuggingFace**: https://huggingface.co/google/owlv2-base-patch16-ensemble

### Option 5: RF-DETR Fine-Tuned (Best Accuracy - Requires Annotation)

- **What**: SOTA transformer detector (ICLR 2026) fine-tuned on domain-specific data
- **Why**: DINOv2 backbone converges fast with small datasets (~200-500 annotated frames); 60.5 mAP on COCO
- **Install**: `pip install rfdetr`
- **Models**:
  - `RFDETRBase` — balanced (resolution 560)
  - `RFDETRLarge` — highest accuracy (resolution 728)
- **Expected gain**: Best possible accuracy for this specific domain
- **Speed**: ~25 FPS on T4 GPU
- **Effort**: High (1-2 weeks, includes annotation)
- **HuggingFace task**: `object-detection`
- **Reference**: https://github.com/roboflow/rf-detr

---

## Recommended Architecture: Hybrid Two-Stage Pipeline

```
Frame Input
    |
    v
+---------------------------+
| Stage 1: YOLO26n + SAHI   |  <-- Every frame (~5-10 FPS)
| (fast bulk detection)      |
+---------------------------+
    |
    | detected objects + frames with low/no detections
    v
+---------------------------+
| Stage 2: Grounding DINO   |  <-- Voting verification (~2-3 FPS)
| or Florence-2             |
| (text-prompted zero-shot) |
+---------------------------+
    |
    v
Merged detections with temporal filtering
```

### Why Two Stages?

| Stage | Model | Purpose | Speed | Accuracy |
|-------|-------|---------|-------|----------|
| Primary (every frame) | YOLO26n + SAHI | Fast detection with sliced inference | ~5-10 FPS | Moderate |
| Voting/verification | Grounding DINO or Florence-2 | Catch what YOLO misses | ~2-3 FPS | High |

### Benefits
- YOLO speed preserved for real-time bulk processing
- Zero-shot model catches missed books, phones, cups during verification
- No retraining needed for initial deployment
- Can progressively fine-tune RF-DETR with accumulated annotations

---

## Implementation Phases

### Phase 1: SAHI Integration (Week 1)
- [ ] Enable SAHI for primary YOLO detection (config exists, needs activation)
- [ ] Use YOLO26l for SAHI passes on person ROI crops
- [ ] Tune slice size (320x320 or 416x416) and overlap (0.2-0.3)
- [ ] Benchmark on n_5_violations_frames

### Phase 2: Zero-Shot Voting Model (Week 2)
- [ ] Benchmark Grounding DINO vs Florence-2 vs OWLv2 on sample frames
- [ ] Select best model based on accuracy/speed tradeoff
- [ ] Integrate as voting verification backend (replace or supplement YOLO26l voting)
- [ ] Text prompts: `"book . cell phone . mobile phone . bag . backpack . cup . bottle . water bottle"`

### Phase 3: Evaluation & Tuning (Week 3)
- [ ] Run full pipeline on n_5_violations_frames (90 frames)
- [ ] Compare detection rates: YOLO-only vs SAHI+YOLO vs Hybrid
- [ ] Tune confidence thresholds for zero-shot model
- [ ] Evaluate false positive rate (control panel knobs detected as phone, etc.)

### Phase 4: Optional Fine-Tuning (Week 4+)
- [ ] Annotate 200-500 frames from multiple cabin camera angles
- [ ] Fine-tune RF-DETR on domain-specific data
- [ ] Replace/augment zero-shot model with fine-tuned model

---

## Test Frames

- Location: `/Users/satishvanga/Desktop/Practice/n_5_violations_frames/`
- Frame count: 90 frames
- Source: IPCamera 02, overhead CCTV
- Timestamp: 2025-11-22, 08:24 - 08:27
- Resolution: 1280x720 (estimated)
- Objects present: book/logbook, bag/backpack, cell phone, cup, water bottle

---

## Dependencies Summary

| Model | Package | GPU Required | Min Python |
|-------|---------|--------------|------------|
| SAHI | `sahi` | No (uses existing YOLO) | 3.8+ |
| Grounding DINO | `groundingdino` or `transformers` | Recommended | 3.10+ |
| Florence-2 | `transformers>=4.36` | Recommended | 3.10+ |
| OWLv2 | `transformers>=4.36` | Recommended | 3.10+ |
| RF-DETR | `rfdetr` | Yes | 3.10+ |
