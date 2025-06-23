import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from method import run_ga, SA, run_grid_search, greedy_search, astar_search

# === 路径设置 ===
image_folder = r"D:\python\车牌\.venv\images"
model_path = r"D:\python\车牌\.venv\runs\detect\train22\weights\best.pt"
model = YOLO(model_path)

# === 参数设置 ===
kernel = [3, 5, 7]
results = {
    'GA': [],
    'SA': [],
    'Grid': [],
    'Greedy': [],
    'A*': []
}

def eval_contour(individual, gray_roi):
    kernel_index, low, high = individual
    if low >= high:
        return -1
    k = kernel[int(kernel_index)]
    blurred = cv2.GaussianBlur(gray_roi, (k, k), 0)
    edges = cv2.Canny(blurred, int(low), int(high))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

# === 主函数：对文件夹中的每张图像进行评估 ===
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))]

for img_path in image_paths:
    print(f"\nProcessing: {img_path}")
    image = cv2.imread(img_path)
    results_yolo = model.predict(source=image, conf=0.25, save=False)
    boxes = results_yolo[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        print("YOLO未检测到车牌，跳过")
        continue

    x1, y1, x2, y2 = map(int, boxes[0])
    roi = image[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    def local_eval(ind): return eval_contour(ind, gray_roi)

    # 每种算法运行一次
    for name, method in {
        'GA': lambda: run_ga(local_eval, kernel),
        'SA': lambda: SA(local_eval, kernel, low_range=(10, 90), high_range=(100, 500)),
        'Grid': lambda: run_grid_search(local_eval, kernel, (10, 90), (100, 500)),
        'Greedy': lambda: greedy_search(local_eval, kernel)[1:],  # 忽略 kernel 位置
        'A*': lambda: astar_search(local_eval, kernel)[1:]
    }.items():
        try:
            start = time.time()
            param = method()
            score = local_eval(param)
            duration = time.time() - start
            results[name].append((score, duration))
            print(f"{name:7} 轮廓: {score:>3}  耗时: {duration:.3f}s")
        except Exception as e:
            results[name].append((0, 0))
            print(f"{name:7} 失败: {e}")

# === 汇总平均表现 ===
print("\n=== 平均表现对比（共{}张图） ===".format(len(image_paths)))
for name in results:
    if results[name]:
        scores, durations = zip(*results[name])
        print(f"{name:<7} 平均轮廓数: {np.mean(scores):>5.2f}  |  平均耗时: {np.mean(durations):.3f}s")
    else:
        print(f"{name:<7} 无结果")
