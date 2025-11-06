#!/usr/bin/env python3
"""Debug OCR detection on a single frame."""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import ocrmac

video_path = "/Users/alexandervyhmeister/Movies/2025-11-04 08-42-51.mp4"

# Extract first frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video")
    exit(1)

print(f"Frame shape: {frame.shape}")

# Focus on top 20%
height = frame.shape[0]
top_region = frame[0:int(height * 0.2), :]
print(f"Top region shape: {top_region.shape}")

# Convert BGR to RGB
frame_rgb = cv2.cvtColor(top_region, cv2.COLOR_BGR2RGB)

# Upscale 2x
h, w = frame_rgb.shape[:2]
frame_rgb = cv2.resize(frame_rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
print(f"Upscaled shape: {frame_rgb.shape}")

# Save for inspection
pil_image = Image.fromarray(frame_rgb)
pil_image.save("/tmp/frame_top_region.png")
print("Saved frame to /tmp/frame_top_region.png")

# Enhance contrast
enhancer = ImageEnhance.Contrast(pil_image)
enhanced = enhancer.enhance(2.0)
enhanced.save("/tmp/frame_enhanced.png")
print("Saved enhanced frame to /tmp/frame_enhanced.png")

# Run OCR on both
print("\n=== OCR on original top region ===")
try:
    annotations = ocrmac.OCR(pil_image).recognize(recognition_level="accurate")
    print(f"Found {len(annotations)} text annotations")
    for i, ann in enumerate(annotations, 1):
        text = ann.get("text", "")
        print(f"  {i}. '{text}'")
except Exception as e:
    print(f"OCR failed: {e}")

print("\n=== OCR on enhanced ===")
try:
    annotations = ocrmac.OCR(enhanced).recognize(recognition_level="accurate")
    print(f"Found {len(annotations)} text annotations")
    for i, ann in enumerate(annotations, 1):
        text = ann.get("text", "")
        print(f"  {i}. '{text}'")
except Exception as e:
    print(f"OCR failed: {e}")

print("\nOpen /tmp/frame_top_region.png and /tmp/frame_enhanced.png to inspect")
