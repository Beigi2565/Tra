---
tags:
- tinyml
- edge-ai
- object-detection
- iot
- automotive
base_model:
- google/mobilenet_v2_1.0_224
---

# tra-base-v1

`tra-base-v1` is a lightweight, TinyML-focused model designed for vehicle counting and localization through density map estimation. This is a base version (v1) suitable for direct deployment or further fine-tuning on custom datasets.

---

## Model Specifications & Resource Usage

- **Target Applications**: Traffic monitoring, vehicle counting on edge devices.
- **Memory Footprint**: 
  - Maximum RAM consumption during 32-bit inference is approximately **5.4 MB**.
  - Suitable for microcontrollers and embedded boards with at least **8 MB of RAM**.
- **Accuracy**: Around **75% to 80%** under recommended deployment conditions.
- **Optimal Deployment Conditions**: For best results, the camera should be positioned at an angle of **45 to 60 degrees** relative to the road.

---

## Visualizations & Test Results

The following test samples demonstrate the model's performance, resource usage, and predicted density maps. The images are loaded from the `results/` folder:

<div align="center">
  <img src="results/traffic-test.png" style="border-radius: 15px; margin: 10px; max-width: 70%;" />
</div>
<div align="center">
  <img src="results/traffic-test2.png" style="border-radius: 15px; margin: 10px; max-width: 70%;" />
</div>
<div align="center">
  <img src="results/traffic-test3.png" style="border-radius: 15px; margin: 10px; max-width: 70%;" />
</div>
<div align="center">
  <img src="results/traffic-test4.png" style="border-radius: 15px; margin: 10px; max-width: 70%;" />
</div>

---

## Limitations

- **High Density Traffic**: As shown in the test images, the model may struggle to provide highly precise counts in congested areas with high vehicle density or significant overlapping.
- **Base Version**: This is a base model; while it performs reasonably well given its extremely low memory usage, additional fine-tuning may be required for complex environments.

---

## Installation & Requirements

To install the required libraries and prepare the environment, run the following commands:

```bash
pip install onnxruntime opencv-python-headless scipy matplotlib psutil --quiet
```

```python
from huggingface_hub import snapshot_download
import os

repo_id = "realAABeigi/tra-base-1"

print(f"[INFO] Downloading all files from {repo_id} to root...")

try:
    # This will download all files from the repo and place them in the current directory
    snapshot_download(
        repo_id=repo_id,
        local_dir="./",
        local_dir_use_symlinks=False
    )
    print("[SUCCESS] All files downloaded to root directory.")
except Exception as e:
    print(f"[ERROR] Failed to download: {e}")
```

```python
!pip install onnxruntime opencv-python-headless scipy matplotlib psutil --quiet

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import gc
import tracemalloc
import time
import psutil
import onnxruntime as ort

THRESHOLD = 0.4
IMG_SIZE = 192
GRID_SIZE = 48
ONNX_PATH = "tra-base.onnx"
DATA_PATH = "tra-base.onnx.data"
TEST_DIR = "/content/Test"

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

try:
    model_size = os.path.getsize(ONNX_PATH)
    if os.path.exists(DATA_PATH):
        model_size += os.path.getsize(DATA_PATH)

    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    def preprocess(img_path):
        orig_img = cv2.imread(img_path)
        if orig_img is None: return None, None
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_data = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)
        return img_data, img_rgb

    def run_onnx_cpu_inference(img_path):
        gc.collect()
        tracemalloc.start()

        mem_before = get_process_memory()
        start_time = time.time()

        img_data, img_rgb = preprocess(img_path)
        if img_data is None: return

        outputs = session.run(None, {input_name: img_data})

        mem_after = get_process_memory()
        inference_time = (time.time() - start_time) * 1000

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        heatmap = outputs[0].squeeze()
        data_max = maximum_filter(heatmap, size=3)
        maxima = (heatmap == data_max) & (heatmap > THRESHOLD)
        y_coords, x_coords = np.where(maxima)

        system_delta = mem_after - mem_before
        total_footprint_kb = (model_size + peak + abs(system_delta)) / 1024

        print(f"\n--- Image: {os.path.basename(img_path)} ---")
        print(f"Latency: {inference_time:.2f}ms")
        print(f"System Memory Change: {system_delta/1024:.2f} KB")
        print(f"Total RAM Footprint (Est): {total_footprint_kb:.2f} KB")

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        display_img = cv2.resize(img_rgb, (384, 384))
        for y, x in zip(y_coords, x_coords):
            cx, cy = int(x * (384/GRID_SIZE)), int(y * (384/GRID_SIZE))
            cv2.circle(display_img, (cx, cy), 6, (255, 0, 0), -1)
        plt.imshow(display_img)
        plt.title(f"Cars: {len(y_coords)} | Time: {inference_time:.1f}ms")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title(f"Total RAM: {total_footprint_kb:.1f} KB")
        plt.axis('off')
        plt.show()

    if os.path.exists(TEST_DIR):
        image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in image_files:
            run_onnx_cpu_inference(os.path.join(TEST_DIR, img_file))
    else:
        print("Folder Test not found.")

except Exception as e:
    print(f"[ERROR]: {e}")
```