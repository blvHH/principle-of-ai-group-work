import cv2
import numpy as np
from ultralytics import YOLO
import random
import math
from matplotlib import pyplot as plt
from deap import base, creator, tools, algorithms
import heapq
from sklearn.cluster import KMeans


#遗传
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

#模拟退火
def SA(eval_contour,kernel,low_range,high_range,initial_T=200.0,max_iter=2000,alpha=0.96) :
    current=[random.randint(0,len(kernel)-1),
             random.randint(*low_range),
             random.randint(*high_range)]
    current_best=current[:]
    best_score = eval_contour(current_best)
    T=initial_T
    for i in range(max_iter) :
        candidate=current[:]
        index=random.randint(0,2)
        if index == 0:
            candidate[0] = random.randint(0, len(kernel) - 1)
        elif index == 1:
            candidate[1] = np.clip(candidate[1] + random.randint(-10, 10), *low_range)
        else:
            candidate[2] = np.clip(candidate[2] + random.randint(-10, 10), *high_range)
        new_score=eval_contour(candidate)
        if new_score>best_score :
            best_score=new_score
            current_best=candidate[:]
            current = candidate[:]
        if new_score<best_score :
            p=math.exp((new_score-best_score)/T)
            r=random.uniform(0,1)
            if p>r :
                best_score=new_score
                current = candidate[:]
        T=T*alpha
        if T < 1e-3:
            break
    return current_best



#网格搜索
def run_grid_search(eval_contour,kernel,low_range,high_range):
    best_params = None
    best_score = -1

    for k_index, k in enumerate(kernel):
        for low in range(low_range[0], low_range[1] + 1, 10):  # 步长可调
            for high in range(high_range[0], high_range[1] + 1, 10):
                if low >= high:
                    continue
                score = eval_contour([k_index, low, high])
                if score > best_score:
                    best_score = score
                    best_params = [k_index, low, high]

    return best_params

#A*
def astar_search(eval_contour,kernel):
    visited = set()
    heap = []

    # 初始状态：kernel index=1, low=50, high=150
    start = (1, 50, 150)
    heapq.heappush(heap, (0, start))

    best_score = -1
    best_state = start

    while heap:
        _, (k_idx, low, high) = heapq.heappop(heap)
        state_key = (k_idx, low, high)
        if state_key in visited:
            continue
        visited.add(state_key)

        score = eval_contour((k_idx, low, high))

        if score > best_score:
            best_score = score
            best_state = (k_idx, low, high)

        # 生成邻居状态
        for nk_idx in [k_idx - 1, k_idx, k_idx + 1]:
            if 0 <= nk_idx < len(kernel):
                for dl in [-10, 0, 10]:
                    for dh in [-10, 0, 10]:
                        if dl == dh == 0:
                            continue
                        nlow = low + dl
                        nhigh = high + dh
                        if 10 <= nlow < 100 and 100<= nhigh < 500:
                            next_state = (nk_idx, nlow, nhigh)
                            h =  1 / (eval_contour(next_state) + 1e-5)  # 负的评分用于最小堆排序
                            heapq.heappush(heap, (h, next_state))
    final=[]
    final.append(best_state[0])
    final.append(best_state[1])
    final.append(best_state[2])

    return kernel[best_state[0]],best_state[1],best_state[2]

#贪心
def greedy_search(eval_contour,kernel):
    current_state = (1, 50, 150)  # 初始值
    best_score = eval_contour(current_state)

    while True:
        found_better = False
        k_idx, low, high = current_state
        neighbors = []

        # 生成所有邻居
        for nk_idx in [k_idx - 1, k_idx, k_idx + 1]:
            if 0 <= nk_idx < len(kernel):
                for dl in [-10, 0, 10]:
                    for dh in [-10, 0, 10]:
                        if dl == dh == 0:
                            continue
                        nlow = low + dl
                        nhigh = high + dh
                        if 10 <= nlow < 100 and 100<= nhigh < 500:
                            neighbors.append((nk_idx, nlow, nhigh))

        # 在所有邻居中选择评分最高的
        best_neighbor = current_state
        for neighbor in neighbors:
            score = eval_contour(neighbor)
            if score > best_score:
                best_score = score
                best_neighbor = neighbor
                found_better = True

        if not found_better:
            break  # 找不到更好的了，停止
        current_state = best_neighbor
    final=[]
    final.append(current_state[0])
    final.append(current_state[1])
    final.append(current_state[2])
    return kernel[current_state[0]],current_state[1],current_state[2]


#Knn
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

    # 可视化
    knn_result = roi.copy()
    cv2.drawContours(knn_result, final_contours, -1, (0, 100, 255), 1)
    plt.figure(figsize=(6, 3))
    plt.imshow(cv2.cvtColor(knn_result, cv2.COLOR_BGR2RGB))
    plt.title(f"{label} - KNN密度裁剪 (面积≥{int(min_cover_ratio*100)}%)")
    plt.axis("off")
    plt.show()

    cropped = roi[y:y + h, x:x + w]
    return cv2.resize(cropped, (500, 200), interpolation=cv2.INTER_CUBIC)

