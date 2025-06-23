import cv2
import numpy as np
from ultralytics import YOLO
import random
import math
from matplotlib import pyplot as plt
from deap import base, creator, tools, algorithms
from method import run_ga,SA,run_grid_search,greedy_search,astar_search,crop_by_contours_knn
import heapq
from sklearn.cluster import KMeans

model=YOLO(r"D:\python\车牌\.venv\runs\detect\train22\weights\best.pt")
image_path=r"C:\Users\21352\Desktop\OIP (5).webp"
image=cv2.imread(image_path)
results=model.predict(source=image,conf=0.25,save=False)
boxes=results[0].boxes.xyxy.cpu().numpy()

if len(boxes) ==0 :
    raise ValueError("No object")
x1,y1,x2,y2=map(int,boxes[0])
roi=image[y1:y2,x1:x2]
gray_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

kernel=[3,5,7]
low_range=(10,100)
high_range=(101,500)

def eval_contour(individual) :
    kernel_index,low,high=individual
    if low>=high :
        return -1
    k=kernel[int(kernel_index)]
    blurred=cv2.GaussianBlur(gray_roi,(k,k),0)
    edges=cv2.Canny(blurred,int(low),int(high))
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

# ========== 手动参数 ==========
manual_kernel = 5
manual_low = 50
manual_high = 150
blur_manual = cv2.GaussianBlur(gray_roi, (manual_kernel, manual_kernel), 0)
edges_manual = cv2.Canny(blur_manual, manual_low, manual_high)
contours_manual, _ = cv2.findContours(edges_manual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
manual_result = roi.copy()
cv2.drawContours(manual_result, contours_manual, -1, (0, 255, 0), 1)

# ========== GA 参数 ==========
best_ga = run_ga(eval_contour,kernel)
k_ga = kernel[int(best_ga[0])]
low_ga = int(best_ga[1])
high_ga = int(best_ga[2])
blur_ga = cv2.GaussianBlur(gray_roi, (k_ga, k_ga), 0)
edges_ga = cv2.Canny(blur_ga, low_ga, high_ga)
contours_ga, _ = cv2.findContours(edges_ga, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ga_result = roi.copy()
cv2.drawContours(ga_result, contours_ga, -1, (255, 0, 0), 1)

# ========== SA 参数 ==========
best_sa = SA(eval_contour, kernel, low_range, high_range)
k_sa = kernel[int(best_sa[0])]
low_sa = int(best_sa[1])
high_sa = int(best_sa[2])
blur_sa = cv2.GaussianBlur(gray_roi, (k_sa, k_sa), 0)
edges_sa = cv2.Canny(blur_sa, low_sa, high_sa)
contours_sa, _ = cv2.findContours(edges_sa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sa_result = roi.copy()
cv2.drawContours(sa_result, contours_sa, -1, (0, 255, 255), 1)

best_grid = run_grid_search(eval_contour, kernel, low_range, high_range)
# ========== Grid Search 结果 ==========
k_grid = kernel[int(best_grid[0])]
low_grid = int(best_grid[1])
high_grid = int(best_grid[2])
blur_grid = cv2.GaussianBlur(gray_roi, (k_grid, k_grid), 0)
edges_grid = cv2.Canny(blur_grid, low_grid, high_grid)
contours_grid, _ = cv2.findContours(edges_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
grid_result = roi.copy()
cv2.drawContours(grid_result, contours_grid, -1, (255, 0, 255), 1)  # 紫色

# ========== Greedy Search ==========
k_greedy, low_greedy, high_greedy = greedy_search(eval_contour,kernel)
blur_greedy = cv2.GaussianBlur(gray_roi, (k_greedy, k_greedy), 0)
edges_greedy = cv2.Canny(blur_greedy, low_greedy, high_greedy)
contours_greedy, _ = cv2.findContours(edges_greedy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
greedy_result = roi.copy()
cv2.drawContours(greedy_result, contours_greedy, -1, (255, 255, 0), 1)  # 黄色

# ========== A* Search ==========
k_astar, low_astar, high_astar = astar_search(eval_contour,kernel)
blur_astar = cv2.GaussianBlur(gray_roi, (k_astar, k_astar), 0)
edges_astar = cv2.Canny(blur_astar, low_astar, high_astar)
contours_astar, _ = cv2.findContours(edges_astar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
astar_result = roi.copy()
cv2.drawContours(astar_result, contours_astar, -1, (0, 128, 255), 1)  # 橙蓝

fig_all, axs_all = plt.subplots(2, 3, figsize=(18, 10))

axs_all[0, 0].imshow(cv2.cvtColor(manual_result, cv2.COLOR_BGR2RGB))
axs_all[0, 0].set_title(f"Manual ({len(contours_manual)})")
axs_all[0, 0].axis("off")

axs_all[0, 1].imshow(cv2.cvtColor(ga_result, cv2.COLOR_BGR2RGB))
axs_all[0, 1].set_title(f"GA ({len(contours_ga)})")
axs_all[0, 1].axis("off")

axs_all[0, 2].imshow(cv2.cvtColor(sa_result, cv2.COLOR_BGR2RGB))
axs_all[0, 2].set_title(f"SA ({len(contours_sa)})")
axs_all[0, 2].axis("off")

axs_all[1, 0].imshow(cv2.cvtColor(grid_result, cv2.COLOR_BGR2RGB))
axs_all[1, 0].set_title(f"Grid ({len(contours_grid)})")
axs_all[1, 0].axis("off")

axs_all[1, 1].imshow(cv2.cvtColor(greedy_result, cv2.COLOR_BGR2RGB))
axs_all[1, 1].set_title(f"Greedy ({len(contours_greedy)})")
axs_all[1, 1].axis("off")

axs_all[1, 2].imshow(cv2.cvtColor(astar_result, cv2.COLOR_BGR2RGB))
axs_all[1, 2].set_title(f"A* ({len(contours_astar)})")
axs_all[1, 2].axis("off")

plt.tight_layout()
plt.show()


# ========== 四种方法裁剪 ==========

def crop_by_contours(contours, label):
    if len(contours) == 0:
        print(f"[{label}] 无法裁剪：没有轮廓")
        return np.zeros((200, 500, 3), dtype=np.uint8)  # 返回空图像占位
    all_pts = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_pts)
    cropped = roi[y:y + h, x:x + w]
    return cv2.resize(cropped, (500, 200), interpolation=cv2.INTER_CUBIC)

crop_manual = crop_by_contours(contours_manual, "Manual")
crop_ga = crop_by_contours(contours_ga, "GA")
crop_sa = crop_by_contours(contours_sa, "SA")
crop_grid = crop_by_contours(contours_grid, "Grid")
crop_greedy = crop_by_contours(contours_greedy, "Greedy")
crop_astar = crop_by_contours(contours_astar, "A*")

print("\n========= 参数对比 =========")
print(f"[Manual] Kernel: {manual_kernel}, Low: {manual_low}, High: {manual_high}")
print(f"[GA]     Kernel: {k_ga}, Low: {low_ga}, High: {high_ga}")
print(f"[SA]     Kernel: {k_sa}, Low: {low_sa}, High: {high_sa}")
print(f"[Grid]   Kernel: {k_grid}, Low: {low_grid}, High: {high_grid}")
print(f"[Greedy] Kernel: {k_greedy}, Low: {low_greedy}, High: {high_greedy}")
print(f"[A*]     Kernel: {k_astar}, Low: {low_astar}, High: {high_astar}")
print("============================\n")


fig_crop, axs_crop = plt.subplots(2, 3, figsize=(18, 6))

titles = ["Manual", "GA", "SA", "Grid", "Greedy", "A*"]
crops = [crop_manual, crop_ga, crop_sa, crop_grid, crop_greedy, crop_astar]

for i in range(6):
    row, col = divmod(i, 3)
    axs_crop[row, col].imshow(cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB))
    axs_crop[row, col].set_title(f"{titles[i]} Cropped")
    axs_crop[row, col].axis("off")

plt.tight_layout()
plt.show()

crop_knn_manual = crop_by_contours_knn(contours_manual,  roi,"Manual")
crop_knn_ga = crop_by_contours_knn(contours_ga,  roi,"GA")
crop_knn_sa = crop_by_contours_knn(contours_sa, roi, "SA")
crop_knn_grid = crop_by_contours_knn(contours_grid,  roi,"Grid")
crop_knn_astar=crop_by_contours_knn(contours_astar,roi,"Astar")
crop_knn_greedy=crop_by_contours_knn(contours_greedy,roi,"Astar")

# ==== 可视化最终的KNN裁剪图 ====
fig_knn, axs_knn = plt.subplots(2, 3, figsize=(12, 6))

axs_knn[0, 0].imshow(cv2.cvtColor(crop_knn_manual, cv2.COLOR_BGR2RGB))
axs_knn[0, 0].set_title("Manual + KNN Cropped")
axs_knn[0, 0].axis("off")

axs_knn[0, 1].imshow(cv2.cvtColor(crop_knn_ga, cv2.COLOR_BGR2RGB))
axs_knn[0, 1].set_title("GA + KNN Cropped")
axs_knn[0, 1].axis("off")

axs_knn[1, 0].imshow(cv2.cvtColor(crop_knn_sa, cv2.COLOR_BGR2RGB))
axs_knn[1, 0].set_title("SA + KNN Cropped")
axs_knn[1, 0].axis("off")

axs_knn[1, 1].imshow(cv2.cvtColor(crop_knn_grid, cv2.COLOR_BGR2RGB))
axs_knn[1, 1].set_title("Grid + KNN Cropped")
axs_knn[1, 1].axis("off")

axs_knn[0, 2].imshow(cv2.cvtColor(crop_knn_astar, cv2.COLOR_BGR2RGB))
axs_knn[0, 2].set_title("A* + KNN Cropped")
axs_knn[0, 2].axis("off")

axs_knn[1, 2].imshow(cv2.cvtColor(crop_knn_greedy, cv2.COLOR_BGR2RGB))
axs_knn[1, 2].set_title("Greedy + KNN Cropped")
axs_knn[1, 2].axis("off")

plt.tight_layout()
plt.show()