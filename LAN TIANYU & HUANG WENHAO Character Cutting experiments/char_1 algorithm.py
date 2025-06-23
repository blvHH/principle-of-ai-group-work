import os
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from method import run_ga, SA, run_grid_search, greedy_search, astar_search
#
# # ========= Settings =========
# model = YOLO(r"D:\python\车牌\.venv\runs\detect\train22\weights\best.pt")
# image_folder = r"D:\python\车牌\.venv\images"
# kernel = [3, 5, 7]
# low_range = (10, 100)
# high_range = (101, 500)
#
# # ========= Evaluation function =========
# def eval_contour(individual, gray_roi):
#     kernel_index, low, high = individual
#     if low >= high:
#         return -1
#     k = kernel[int(kernel_index)]
#     blurred = cv2.GaussianBlur(gray_roi, (k, k), 0)
#     edges = cv2.Canny(blurred, int(low), int(high))
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return len(contours)
#
# # ========= Collect results =========
# results = []
#
# for filename in os.listdir(image_folder):
#     if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
#         continue
#
#     path = os.path.join(image_folder, filename)
#     image = cv2.imread(path)
#     predict = model.predict(source=image, conf=0.25, save=False)
#     boxes = predict[0].boxes.xyxy.cpu().numpy()
#
#     if len(boxes) == 0:
#         print(f"{filename}: No object detected")
#         continue
#
#     x1, y1, x2, y2 = map(int, boxes[0])
#     roi = image[y1:y2, x1:x2]
#     gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#
#     def eval_wrap(ind): return eval_contour(ind, gray_roi)
#
#     # Manual parameters
#     k_manual, low_manual, high_manual = 5, 50, 150
#     blur = cv2.GaussianBlur(gray_roi, (k_manual, k_manual), 0)
#     edges = cv2.Canny(blur, low_manual, high_manual)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     manual_score = len(contours)
#
#     # Algorithms
#     ga_score = eval_wrap(run_ga(eval_wrap, kernel))
#     sa_score = eval_wrap(SA(eval_wrap, kernel, low_range, high_range))
#     grid_score = eval_wrap(run_grid_search(eval_wrap, kernel, low_range, high_range))
#     greedy_k, low_g, high_g = greedy_search(eval_wrap, kernel)
#     greedy_score = eval_wrap([kernel.index(greedy_k), low_g, high_g])
#     astar_k, low_a, high_a = astar_search(eval_wrap, kernel)
#     astar_score = eval_wrap([kernel.index(astar_k), low_a, high_a])
#
#     # Append to results
#     scores = [manual_score, ga_score, sa_score, grid_score, greedy_score, astar_score]
#     avg_score = round(sum(scores) / len(scores), 2)
#
#     results.append({
#         "Image": filename,
#         "Manual": manual_score,
#         "GA": ga_score,
#         "SA": sa_score,
#         "Grid": grid_score,
#         "Greedy": greedy_score,
#         "A*": astar_score,
#         "Average": avg_score
#     })
#
# # ========= Export DataFrame to Excel =========
# df = pd.DataFrame(results)
# df.to_excel("contour_optimization_scores.xlsx", index=False)
# print("✅ Saved to contour_optimization_scores.xlsx")
df = pd.read_excel("轮廓优化结果评分.xlsx")

# 如果没有 "Average" 列，先加上平均值列（可选）
if "Average" not in df.columns:
    df["Average"] = df[["Manual", "GA", "SA", "Grid", "Greedy", "A*"]].mean(axis=1)
# ========= Plot Line Chart =========
methods = ["Manual", "GA", "SA", "Grid", "Greedy", "A*"]

# 计算各方法的平均轮廓数
mean_scores = df[methods].mean()

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(methods, mean_scores, marker='o', linestyle='-', color='darkorange', linewidth=2)
plt.title("Average Contour Count by Optimization Method")
plt.ylabel("Average Contour Count")
plt.xlabel("Method")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("average_contour_lineplot.png")
plt.show()