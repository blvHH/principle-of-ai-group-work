from __future__ import annotations
import cv2, random, time, sys, math, functools
import numpy as np
from typing import List, Tuple, Dict
import 基础 as base

RESTARTS        = 5            
HMS             = 20           
ITERS_STAGE     = 180         
STAGES          = 3           
HMCR_VALUES     = [0.85, 0.92, 0.97]  
PAR_VALUES      = [0.45, 0.35, 0.20]  
BW_FRACTIONS    = [0.40, 0.15, 0.05] 
TIME_LIMIT      = 10.0     
ELITE_COUNT     = 3           
SEED            = 88
random.seed(SEED); np.random.seed(SEED)

class EnhancedFeatureExtractor:
    def __init__(self, plate: np.ndarray):
        self.plate = plate
        self.W = plate.shape[1]
        self.H = plate.shape[0]

        self.edge_map = self._compute_edge_map()
        self.projection = self._compute_projection()
        self.gradient_map = self._compute_gradient_map()
        
    def _compute_edge_map(self) -> np.ndarray:
        edges1 = cv2.Canny(self.plate, 30, 90)
        edges2 = cv2.Canny(self.plate, 50, 150) 

        combined = cv2.bitwise_or(edges1, edges2)

        kernel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        vertical_edges = cv2.filter2D(self.plate.astype(np.float32), -1, kernel_v)
        vertical_edges = np.abs(vertical_edges).astype(np.uint8)
        
        return cv2.bitwise_or(combined, vertical_edges)
    
    def _compute_projection(self) -> np.ndarray:
        binary = (self.plate < 128).astype(np.uint8)
        return np.sum(binary, axis=0)
    
    def _compute_gradient_map(self) -> np.ndarray:
        grad_x = cv2.Sobel(self.plate, cv2.CV_32F, 1, 0, ksize=3)
        return np.abs(grad_x)

@functools.lru_cache(maxsize=8192)
def _enhanced_fitness_cached(key: str, W: int, H: int):
    cuts = list(map(int, key.split(',')))
    plate = _enhanced_fitness_cached.plate
    edge_map = _enhanced_fitness_cached.edge_map
    projection = _enhanced_fitness_cached.projection
    gradient_map = _enhanced_fitness_cached.gradient_map
    
    # 字符区域提取
    chars = [plate[:, cuts[i]:cuts[i+1]] for i in range(7)]
    widths = np.array([c.shape[1] for c in chars])
    
    # 1. 字符识别置信度 (权重: 0.45)
    conf_scores = []
    for char in chars:
        if char.shape[1] > 0:
            conf = base.recognize_one_char_basic(char)[1]
            conf_scores.append(conf)
        else:
            conf_scores.append(0.0)
    avg_confidence = np.mean(conf_scores)
    
    # 2. 边缘对齐度 (权重: 0.20)
    edge_alignment = 0.0
    for i in range(1, 7):
        if cuts[i] < edge_map.shape[1]:
            # 切割线处的垂直边缘强度
            edge_strength = np.sum(edge_map[:, max(0, cuts[i]-1):min(edge_map.shape[1], cuts[i]+2)])
            edge_alignment += edge_strength
    edge_alignment = edge_alignment / (6 * H * 3 * 255)  # 归一化
    
    # 3. 投影谷值对齐 (权重: 0.15)
    projection_score = 0.0
    for i in range(1, 7):
        if cuts[i] < len(projection):
            # 寻找切割点附近的投影最小值
            window = projection[max(0, cuts[i]-3):min(len(projection), cuts[i]+4)]
            if len(window) > 0:
                min_idx = np.argmin(window)
                actual_min_pos = max(0, cuts[i]-3) + min_idx
                # 距离投影谷值越近越好
                distance_penalty = abs(cuts[i] - actual_min_pos) / 7.0
                projection_score += max(0, 1.0 - distance_penalty)
    projection_score = projection_score / 6
    
    # 4. 宽度一致性 (权重: 0.12)
    ideal_width = W / 7
    width_consistency = 1.0 - np.std(widths) / ideal_width
    width_consistency = max(0, width_consistency)
    
    # 5. 字符完整性 (权重: 0.08)
    completeness_penalty = 0
    for width in widths:
        if width < 6:  # 太窄
            completeness_penalty += 0.5
        elif width > W * 0.6:  # 太宽
            completeness_penalty += 0.3
    completeness = max(0, 1.0 - completeness_penalty / 7)
    
    # 综合评分
    total_score = (0.45 * avg_confidence + 
                   0.20 * edge_alignment + 
                   0.15 * projection_score + 
                   0.12 * width_consistency + 
                   0.08 * completeness)
    
    return total_score

def _enhanced_fitness(cuts: List[int], extractor: EnhancedFeatureExtractor) -> float:
    """增强的适应度函数"""
    _enhanced_fitness_cached.plate = extractor.plate
    _enhanced_fitness_cached.edge_map = extractor.edge_map
    _enhanced_fitness_cached.projection = extractor.projection
    _enhanced_fitness_cached.gradient_map = extractor.gradient_map
    
    key = ','.join(map(str, cuts))
    return _enhanced_fitness_cached(key, extractor.W, extractor.H)

def _valid(cuts: List[int]) -> bool:
    """所有字符宽度 ≥1 px 时返回 True"""
    return all(cuts[i + 1] - cuts[i] > 0 for i in range(7))

def _smart_repair(cuts: List[int], W: int, min_gap: int = 4) -> List[int]:
    """智能修复切割线"""
    repaired = cuts.copy()
    
    # 确保递增且满足最小间距
    for i in range(1, 7):
        repaired[i] = max(repaired[i], repaired[i-1] + min_gap)
    
    for i in range(6, 0, -1):
        repaired[i] = min(repaired[i], repaired[i+1] - min_gap)
    
    # 边界约束
    repaired[0] = 0
    repaired[7] = W
    
    return repaired

def _adaptive_initialization(W: int, extractor: EnhancedFeatureExtractor) -> List[List[int]]:
    initializations = []

    uniform_cuts = [0] + [int(i * W / 7) for i in range(1, 7)] + [W]
    initializations.append(uniform_cuts)

    projection = extractor.projection
    if len(projection) > 0:
        valleys = []
        for i in range(1, len(projection) - 1):
            if (projection[i] < projection[i-1] and 
                projection[i] < projection[i+1] and
                projection[i] < np.mean(projection) * 0.7):
                valleys.append(i)
        
        if len(valleys) >= 6:
            valley_scores = []
            for v in valleys:
                depth = min(projection[v-1], projection[v+1]) - projection[v]
                width = 1
                left = v
                while left > 0 and projection[left] <= projection[v] + depth * 0.3:
                    left -= 1
                right = v
                while right < len(projection) - 1 and projection[right] <= projection[v] + depth * 0.3:
                    right += 1
                width = right - left
                
                score = depth * math.log(width + 1)
                valley_scores.append((v, score))

            valley_scores.sort(key=lambda x: x[1], reverse=True)
            selected_valleys = sorted([v[0] for v in valley_scores[:6]])
            
            projection_cuts = [0] + selected_valleys + [W]
            initializations.append(projection_cuts)

    edge_map = extractor.edge_map
    edge_projection = np.sum(edge_map, axis=0)

    peaks = []
    for i in range(1, len(edge_projection) - 1):
        if (edge_projection[i] > edge_projection[i-1] and 
            edge_projection[i] > edge_projection[i+1] and
            edge_projection[i] > np.mean(edge_projection) * 1.2):
            peaks.append(i)
    
    if len(peaks) >= 6:
        peak_scores = [(p, edge_projection[p]) for p in peaks]
        peak_scores.sort(key=lambda x: x[1], reverse=True)
        selected_peaks = sorted([p[0] for p in peak_scores[:6]])
        
        edge_cuts = [0] + selected_peaks + [W]
        initializations.append(edge_cuts)

    for _ in range(HMS - len(initializations)):
        random_cuts = [0] + sorted([
            int(i * W / 7 + random.uniform(-0.25, 0.25) * W / 7) 
            for i in range(1, 7)
        ]) + [W]
        random_cuts = _smart_repair(random_cuts, W)
        initializations.append(random_cuts)
    
    return initializations

def _improvise_harmony(harmony_memory: List[Tuple[List[int], float]], 
                      W: int, HMCR: float, PAR: float, BW: int,
                      extractor: EnhancedFeatureExtractor) -> Tuple[List[int], float]:
    new_harmony = [0] + [0] * 6 + [W]

    for i in range(1, 7):
        if random.random() < HMCR:
            selected_harmony = random.choice(harmony_memory)[0]
            new_harmony[i] = selected_harmony[i]

            if random.random() < PAR:
                if random.random() < 0.4:
                    projection = extractor.projection
                    current_pos = new_harmony[i]

                    search_range = min(BW, 15)
                    start = max(new_harmony[i-1] + 4, current_pos - search_range)
                    end = min(W if i == 6 else new_harmony[i+1] - 4, current_pos + search_range)
                    
                    if start < end and end < len(projection):
                        window = projection[start:end+1]
                        if len(window) > 0:
                            min_idx = np.argmin(window)
                            new_harmony[i] = start + min_idx
                else:
                    new_harmony[i] += random.randint(-BW, BW)
        else:
            if i == 1:
                new_harmony[i] = random.randint(4, W // 7)
            else:
                min_pos = new_harmony[i-1] + 4
                max_pos = W - 4 * (7 - i) if i < 6 else W - 4
                if min_pos >= max_pos:        
                    new_harmony[i] = min_pos if min_pos < W else W - 1
                else:                      
                    new_harmony[i] = random.randint(min_pos, max_pos)

    new_harmony = _smart_repair(new_harmony, W)
    if not _valid(new_harmony):
        return new_harmony, -1.0

    score = _enhanced_fitness(new_harmony, extractor)
    
    return new_harmony, score

def _local_search(cuts: List[int], extractor: EnhancedFeatureExtractor) -> Tuple[List[int], float]:
    best_cuts = cuts.copy()
    best_score = _enhanced_fitness(best_cuts, extractor)

    for scale in [3, 2, 1]:
        improved = True
        while improved:
            improved = False

            for idx in range(1, 7):
                for direction in [-1, 1]:
                    candidate = best_cuts.copy()
                    candidate[idx] += direction * scale
       
                    if (candidate[idx] > candidate[idx-1] + 4 and 
                        candidate[idx] < candidate[idx+1] - 4):
                        
                        score = _enhanced_fitness(candidate, extractor)
                        if score > best_score + 1e-5:
                            best_cuts, best_score = candidate, score
                            improved = True
    
    return best_cuts, best_score

def _elite_learning(harmony_memory: List[Tuple[List[int], float]], 
                   extractor: EnhancedFeatureExtractor) -> List[Tuple[List[int], float]]:
    elite_harmonies = sorted(harmony_memory, key=lambda x: x[1], reverse=True)[:ELITE_COUNT]

    improved_elites = []
    for cuts, score in elite_harmonies:
        improved_cuts, improved_score = _local_search(cuts, extractor)
        improved_elites.append((improved_cuts, improved_score))

    harmony_memory.sort(key=lambda x: x[1])
    
    for i, (improved_cuts, improved_score) in enumerate(improved_elites):
        if improved_score > harmony_memory[i][1]:
            harmony_memory[i] = (improved_cuts, improved_score)
    
    return harmony_memory

def _progress_bar(tag: str, current: int, total: int, extra_info: str = ""):
    bar_length = 35
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    percent = int(100 * current / total)
    
    sys.stdout.write(f"\r{tag} [{bar}] {percent:3d}% {extra_info}")
    sys.stdout.flush()

def search_best(*, img_path: str) -> dict:
    img = cv2.imread(img_path)
    box = base.detect_plate_basic(img)
    if not box:
        print("[HS++] Cannot find license plate")
        return {}
    
    x, y, w, h = box
    W = w
    plate = base.ensure_plate_binary(img[y:y+h, x:x+w])

    extractor = EnhancedFeatureExtractor(plate)
    
    global_best_cuts = None
    global_best_score = -1.0
    start_time = time.time()
    
    print(f"[HS++] Start superior，{RESTARTS} restart，{STAGES} stage")
    
    for restart in range(RESTARTS):
        if time.time() - start_time > TIME_LIMIT:
            break

        initializations = _adaptive_initialization(W, extractor)
        harmony_memory = []
        
        for init_cuts in initializations:
            score = _enhanced_fitness(init_cuts, extractor)
            harmony_memory.append((init_cuts, score))

        harmony_memory.sort(key=lambda x: x[1], reverse=True)

        total_iterations = STAGES * ITERS_STAGE
        current_iteration = 0
        
        for stage in range(STAGES):
            HMCR = HMCR_VALUES[stage]
            PAR = PAR_VALUES[stage]
            BW = int(W * BW_FRACTIONS[stage])
            
            for iteration in range(ITERS_STAGE):
                current_iteration += 1

                new_harmony, new_score = _improvise_harmony(
                    harmony_memory, W, HMCR, PAR, BW, extractor
                )

                worst_idx = min(range(len(harmony_memory)), key=lambda i: harmony_memory[i][1])
                
                if new_score > harmony_memory[worst_idx][1]:
                    harmony_memory[worst_idx] = (new_harmony, new_score)

                if iteration % 30 == 0:
                    harmony_memory = _elite_learning(harmony_memory, extractor)

                current_best = max(harmony_memory, key=lambda x: x[1])
                if current_best[1] > global_best_score:
                    global_best_cuts = current_best[0].copy()
                    global_best_score = current_best[1]

                if current_iteration % 20 == 0:
                    _progress_bar(f"[HS++#{restart+1}]", current_iteration, total_iterations, 
                                f"score={global_best_score:.4f}")
        
        print(f"\n[HS++#{restart+1}] Best score: {global_best_score:.4f}")

    if global_best_cuts:
        print("[HS++] Final search...")
        final_cuts, final_score = _local_search(global_best_cuts, extractor)
        
        if final_score > global_best_score:
            global_best_cuts = final_cuts
            global_best_score = final_score
    
    execution_time = time.time() - start_time
    print(f"[HS++] Superior end，use {execution_time:.2f}s，Best score: {global_best_score:.4f}")
    print(f"[HS++] Best cutting position: {global_best_cuts}")
    
    return {"cuts": global_best_cuts} if global_best_cuts else {}
