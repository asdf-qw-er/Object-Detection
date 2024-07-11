import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 모델 로드
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("Model loaded.")

def detect_objects(image_path, model):
    print(f"Loading image from {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None
    print("Image loaded successfully.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Running object detection...")
    results = model(img_rgb)
    print("Object detection completed.")

    return results, img_rgb

def visualize_detections(results, img_rgb):
    if results is None:
        print("Error: No detection results to visualize")
        return

    print("Rendering detection results...")
    results.render()  # 이미지에 박스 및 레이블을 그립니다.

    img_with_boxes = results.ims[0]  # 결과 이미지 가져오기
    plt.figure(figsize=(12, 8))
    plt.imshow(img_with_boxes)
    plt.axis('off')
    plt.show()
    print("Visualization completed.")

    # 결과 이미지를 파일로 저장
    img_output_path = 'detected_output.jpg'
    Image.fromarray(img_with_boxes).save(img_output_path)
    print(f"Output image saved to {img_output_path}")

# 이미지 경로 설정
image_path = 'img01.jpg'  # 여기에 실제 이미지 경로를 넣으세요.

results, img_rgb = detect_objects(image_path, model)
visualize_detections(results, img_rgb)