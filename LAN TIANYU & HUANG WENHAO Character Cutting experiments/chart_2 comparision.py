import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from method import run_ga, crop_by_contours_knn

# ========== Load model and image ==========
model = YOLO(r"D:\python\车牌\.venv\runs\detect\train22\weights\best.pt")
image_path = r"C:\Users\21352\Desktop\OIP (1).jpg"
image = cv2.imread(image_path)
results = model.predict(source=image, conf=0.25, save=False)
boxes = results[0].boxes.xyxy.cpu().numpy()

if len(boxes) == 0:
    raise ValueError("No object detected by YOLOv8")

# ========== Get ROI ==========
x1, y1, x2, y2 = map(int, boxes[0])
roi = image[y1:y2, x1:x2]
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

def resize_image(img):
    return cv2.resize(img, (500, 200), interpolation=cv2.INTER_CUBIC)

# ========== Step 1: YOLO Cropped ==========
step1 = resize_image(roi)

# ========== Step 2 & 3: Manual Canny + Cropped ==========
manual_kernel, manual_low, manual_high = 5, 50, 150
blur_manual = cv2.GaussianBlur(gray_roi, (manual_kernel, manual_kernel), 0)
edges_manual = cv2.Canny(blur_manual, manual_low, manual_high)
contours_manual, _ = cv2.findContours(edges_manual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
step2 = resize_image(cv2.cvtColor(edges_manual, cv2.COLOR_GRAY2BGR))

if contours_manual:
    all_pts = np.concatenate(contours_manual)
    x_m, y_m, w_m, h_m = cv2.boundingRect(all_pts)
    crop_manual = roi[y_m:y_m + h_m, x_m:x_m + w_m]
    step3 = resize_image(crop_manual)
else:
    step3 = np.zeros((200, 500, 3), dtype=np.uint8)

# ========== Step 4 & 5: GA Optimized Canny + Crop ==========
kernel = [3, 5, 7]
low_range, high_range = (10, 100), (101, 500)

def eval_contour(individual):
    kernel_idx, low, high = individual
    if low >= high:
        return -1
    k = kernel[int(kernel_idx)]
    blurred = cv2.GaussianBlur(gray_roi, (k, k), 0)
    edges = cv2.Canny(blurred, int(low), int(high))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

best_ga = run_ga(eval_contour, kernel)
k_ga = kernel[int(best_ga[0])]
low_ga, high_ga = int(best_ga[1]), int(best_ga[2])

blur_ga = cv2.GaussianBlur(gray_roi, (k_ga, k_ga), 0)
edges_ga = cv2.Canny(blur_ga, low_ga, high_ga)
contours_ga, _ = cv2.findContours(edges_ga, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ga_result = roi.copy()
cv2.drawContours(ga_result, contours_ga, -1, (255, 0, 0), 1)
step4 = resize_image(ga_result)

if contours_ga:
    all_pts = np.concatenate(contours_ga)
    x_g, y_g, w_g, h_g = cv2.boundingRect(all_pts)
    crop_ga = roi[y_g:y_g + h_g, x_g:x_g + w_g]
    step5 = resize_image(crop_ga)
else:
    step5 = np.zeros((200, 500, 3), dtype=np.uint8)

# ========== Step 6: GA + KNN filtered contours ==========
knn_result = roi.copy()
for cnt in contours_ga:
    cv2.drawContours(knn_result, [cv2.convexHull(cnt)], -1, (0, 255, 255), 1)
step6 = resize_image(knn_result)

# ========== Step 7: Final cropped image with GA + KNN ==========
step7 = resize_image(crop_by_contours_knn(contours_ga, roi, "GA"))

# ========== Combine and Save All Results ==========
steps = [step1, step2, step3, step4, step5, step6, step7]
titles = [
    "1. YOLO Cropped",
    "2. Manual Canny Edges",
    "3. Manual Canny Cropped",
    "4. GA-Canny Contours",
    "5. GA-Canny Cropped",
    "6. GA-Canny-KNN Contours",
    "7. GA-Canny-KNN Cropped"
]

plt.figure(figsize=(18, 10))
for i, (img, title) in enumerate(zip(steps, titles)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

# Save only one combined image
plt.tight_layout()
plt.savefig("all_steps.jpg", dpi=300)
plt.show()
