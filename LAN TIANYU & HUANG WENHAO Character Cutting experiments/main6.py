import cv2
import numpy as np
from ultralytics import YOLO
import random
from sklearn.cluster import KMeans

model=YOLO(r"D:\python\车牌\.venv\runs\detect\train22\weights\best.pt")
image_path=r"C:\Users\21352\Desktop\OIP (1).webp"
image=cv2.imread(image_path)
results=model.predict(source=image,conf=0.25,save=False)
boxes=results[0].boxes.xyxy.cpu().numpy()

if len(boxes) ==0 :
    raise ValueError("No object")
x1,y1,x2,y2=map(int,boxes[0])
roi=image[y1:y2,x1:x2]
gray_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

kernel=[3,5,7]

def eval_contour(individual) :
    kernel_index,low,high=individual
    if low>=high :
        return -1
    k=kernel[int(kernel_index)]
    blurred=cv2.GaussianBlur(gray_roi,(k,k),0)
    edges=cv2.Canny(blurred,int(low),int(high))
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def run_ga(eval_contour,kernel,population_size=20,generations=30,mutation_rate=0.2,crossover_rate=0.8):
    # 初始化种群，每个个体是 (kernel_index, low, high)
    def random_individual():
        k_idx = random.randint(0, len(kernel) - 1)
        low = random.randint(10, 100)
        high = random.randint(low + 1, 500)
        return (k_idx, low, high)

    population = [random_individual() for _ in range(population_size)]

    for gen in range(generations):
        # 计算适应度
        scored = [(eval_contour(ind), ind) for ind in population]
        scored.sort(reverse=True)
        population = [ind for _, ind in scored]

        # 选择（保留前一半）
        num_elites = population_size // 2
        next_generation = population[:num_elites]

        # 交叉（Crossover）
        while len(next_generation) < population_size:
            if random.random() < crossover_rate:
                p1 = random.choice(population[:10])  # 精英父代
                p2 = random.choice(population[:10])
                # 简单的基因交叉
                child = (
                    random.choice([p1[0], p2[0]]),  # kernel_index
                    random.choice([p1[1], p2[1]]),  # low
                    random.choice([p1[2], p2[2]])   # high
                )
                # 验证合法性
                if child[1] < child[2]:
                    next_generation.append(child)

        # 变异（Mutation）
        for i in range(len(next_generation)):
            if random.random() < mutation_rate:
                k_idx = random.randint(0, len(kernel) - 1)
                delta_l = random.randint(-15, 15)
                delta_h = random.randint(-15, 15)
                l = max(10, next_generation[i][1] + delta_l)
                h = min(500, next_generation[i][2] + delta_h)
                if l < h:
                    next_generation[i] = (k_idx, l, h)

        population = next_generation

    # 最终结果
    best_score, best_ind = max((eval_contour(ind), ind) for ind in population)
    best=[]
    best.append(best_ind[0])
    best.append(best_ind[1])
    best.append(best_ind[2])
    return best

def contour_similarity(c1, c2):
    try:
        h1 = cv2.HuMoments(cv2.moments(c1)).flatten()
        h2 = cv2.HuMoments(cv2.moments(c2)).flatten()
        return -np.sum(np.abs(np.log(np.abs(h1 + 1e-10)) - np.log(np.abs(h2 + 1e-10))))
    except:
        return -1e9

def crop_by_contours_knn(contours,roi,label="KNN",cluster_count=7, area_thresh=8,min_cover_ratio=0.75):
    if len(contours)==0:
        print("No object")
        return cv2.resize(roi,(500,200),interpolation=cv2.INTER_CUBIC)

    centers=[]
    valid_indices=[]
    areas=[]
    final_contours = []

    #计算质心
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            centers.append([cx, cy])
            valid_indices.append(i)
            areas.append(cv2.contourArea(cnt))

    #无质心则返回原图
    if len(centers)==0:
        return cv2.resize(roi,(500,200),interpolation=cv2.INTER_CUBIC)

    if len(centers)<cluster_count:
        cluster_count=max(1,len(centers))

    centers=np.array(centers)
    kmeans=KMeans(n_clusters=cluster_count,n_init=10)
    labels=kmeans.fit_predict(centers)
    cluster_dict = {}
    for i, idx in enumerate(valid_indices):
        cluster_id = labels[i]
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        if areas[i] >= area_thresh:
            cluster_dict[cluster_id].append(contours[idx])

    cluster_scores = []
    for cid, cnts in cluster_dict.items():
        if len(cnts) < 1:
            continue
        pts = []
        for cnt in cnts:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                pts.append([cx, cy])
        if len(pts) < 2:
            continue
        pts = np.array(pts)
        std_dev = np.std(np.linalg.norm(pts - np.mean(pts, axis=0), axis=1)) + 1e-5
        similarities = []
        for i in range(len(cnts)):
            for j in range(i + 1, len(cnts)):
                sim = contour_similarity(cnts[i], cnts[j])
                similarities.append(sim)

        if len(similarities) > 0:
            avg_similarity = np.mean(similarities)
            score = avg_similarity
        else:
            score = -1e9

        cluster_scores.append((score, cid, cnts))

    cluster_scores.sort(reverse=True)

    roi_area = roi.shape[0] * roi.shape[1]
    selected_contours = []
    for i in range(1,len(cluster_scores)+1) :
        cluster=cluster_scores[:i]
        current_contours=[]
        for _,_,contours in cluster :
            current_contours.extend(contours)
        if len(current_contours)==0 :
            continue
        points=np.concatenate(current_contours)
        x,y,w,h=cv2.boundingRect(points)
        area=w*h
        if area/roi_area>=min_cover_ratio :
            final_contours=current_contours
            break
    if len(final_contours) == 0 and cluster_scores:
        merged = []
        for _, _, cnts in cluster_scores:
            merged.extend(cnts)
            if len(merged) > 0:
                all_pts = np.concatenate(merged)
                x, y, w, h = cv2.boundingRect(all_pts)
                crop_area = w * h
                if crop_area / roi_area >= min_cover_ratio:
                    final_contours= merged
                    break

    # 如果还不够，就全选
    if len(final_contours) == 0 and len(cluster_dict) > 0:
        final_contours= [item for sublist in cluster_dict.values() for item in sublist]
        if final_contours:
            all_pts = np.concatenate(final_contours)
            x, y, w, h = cv2.boundingRect(all_pts)

    if len(final_contours) == 0:
        print(f"[{label}-KNN] 最终仍无法满足要求，返回空图")
        return cv2.resize(roi,(500,200),interpolation=cv2.INTER_CUBIC)

    cropped = roi[y:y + h, x:x + w]
    return cv2.resize(cropped, (500, 200), interpolation=cv2.INTER_CUBIC)

best_ga = run_ga(eval_contour,kernel)
k_ga = kernel[int(best_ga[0])]
low_ga = int(best_ga[1])
high_ga = int(best_ga[2])
blur_ga = cv2.GaussianBlur(gray_roi, (k_ga, k_ga), 0)
edges_ga = cv2.Canny(blur_ga, low_ga, high_ga)
contours_ga, _ = cv2.findContours(edges_ga, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

crop_knn_ga = crop_by_contours_knn(contours_ga,  roi,"GA")

cv2.imshow("Cropped License Plate (GA+KNN)", crop_knn_ga)
cv2.waitKey(0)
cv2.destroyAllWindows()