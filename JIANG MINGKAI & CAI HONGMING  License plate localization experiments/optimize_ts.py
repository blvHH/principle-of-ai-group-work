from __future__ import annotations
import cv2, random, time, sys, math, functools, collections
import numpy as np
from typing import List, Tuple, Dict, Set, Deque
import 基础 as base

MAX_ITERS       = 700        
TIME_LIMIT      = 10.0        
TABU_MIN        = 15       
TABU_MAX        = 40          
TABU_DYNAMIC    = True    
ASPIRATION      = True  
DIVERSIFY_GAP   = 100       
INTENSIFY_GAP   = 200       
STALL_GAP       = 80    
SEED            = 2025
random.seed(SEED); np.random.seed(SEED)

# ===== 增强工具函数 =====
class EnhancedFeatureExtractor:
    """增强的特征提取器"""
    
    def __init__(self, plate: np.ndarray):
        self.plate = plate
        self.W = plate.shape[1]
        self.H = plate.shape[0]
        
        # 预计算特征
        self.edge_map = self._compute_edge_map()
        self.projection = self._compute_projection()
        self.gradient_map = self._compute_gradient_map()
        
    def _compute_edge_map(self) -> np.ndarray:
        """计算边缘图"""
        # 多尺度边缘检测
        edges1 = cv2.Canny(self.plate, 30, 90)
        edges2 = cv2.Canny(self.plate, 50, 150) 
        
        # 组合边缘
        combined = cv2.bitwise_or(edges1, edges2)
        
        # 垂直边缘增强（字符边界）
        kernel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        vertical_edges = cv2.filter2D(self.plate.astype(np.float32), -1, kernel_v)
        vertical_edges = np.abs(vertical_edges).astype(np.uint8)
        
        return cv2.bitwise_or(combined, vertical_edges)
    
    def _compute_projection(self) -> np.ndarray:
        """计算水平投影"""
        binary = (self.plate < 128).astype(np.uint8)
        return np.sum(binary, axis=0)
    
    def _compute_gradient_map(self) -> np.ndarray:
        """计算梯度图"""
        grad_x = cv2.Sobel(self.plate, cv2.CV_32F, 1, 0, ksize=3)
        return np.abs(grad_x)

@functools.lru_cache(maxsize=8192)
def _enhanced_fitness_cached(key: str, W: int, H: int):
    """增强的缓存适应度函数"""
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

def _generate_neighbors(cuts: List[int], W: int, step: int, 
                       extractor: EnhancedFeatureExtractor) -> List[Tuple[List[int], float]]:
    neighbors = []

    for idx in range(1, 7):
        for delta in [-step, step]:
            neighbor = cuts.copy()
            neighbor[idx] += delta

            neighbor = _smart_repair(neighbor, W)

            score = _enhanced_fitness(neighbor, extractor)
            neighbors.append((neighbor, score))

    for idx in range(1, 4):
        for delta in [-step, step]:
            neighbor = cuts.copy()
            neighbor[idx] += delta
            neighbor[7-idx] -= delta

            neighbor = _smart_repair(neighbor, W)

            score = _enhanced_fitness(neighbor, extractor)
            neighbors.append((neighbor, score))

    projection = extractor.projection
    for idx in range(1, 7):
        current_pos = cuts[idx]

        search_range = min(step * 2, 15)
        start = max(cuts[idx-1] + 4, current_pos - search_range)
        end = min(cuts[idx+1] - 4, current_pos + search_range)
        
        if start < end and end < len(projection):
            window = projection[start:end+1]
            if len(window) > 0:
                min_idx = np.argmin(window)
                new_pos = start + min_idx
                
                if new_pos != current_pos:
                    neighbor = cuts.copy()
                    neighbor[idx] = new_pos

                    score = _enhanced_fitness(neighbor, extractor)
                    neighbors.append((neighbor, score))
    
    return neighbors

def _hash_cuts(cuts: List[int]) -> str:
    return ','.join(map(str, cuts[1:7]))

def _adaptive_initialization(W: int, extractor: EnhancedFeatureExtractor) -> List[int]:
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
            
            if len(selected_valleys) == 6:
                return [0] + selected_valleys + [W]

    return [0] + [int(i * W / 7) for i in range(1, 7)] + [W]

def _intensification(cuts: List[int], extractor: EnhancedFeatureExtractor) -> Tuple[List[int], float]:
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

def _diversification(cuts: List[int], W: int, extractor: EnhancedFeatureExtractor) -> List[int]:
    perturbed = cuts.copy()
    for i in range(1, 7):
        perturb_range = int(W * 0.15)  
        perturbed[i] += random.randint(-perturb_range, perturb_range)
    
    perturbed = _smart_repair(perturbed, W)

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
        
        if len(selected_peaks) == 6:
            edge_cuts = [0] + selected_peaks + [W]

            perturbed_score = _enhanced_fitness(perturbed, extractor)
            edge_score = _enhanced_fitness(edge_cuts, extractor)
            
            if edge_score > perturbed_score:
                return edge_cuts
    
    return perturbed

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
        print("[TS++] Cannot find license plate")
        return {}
    
    x, y, w, h = box
    W = w
    plate = base.ensure_plate_binary(img[y:y+h, x:x+w])

    extractor = EnhancedFeatureExtractor(plate)

    current_cuts = _adaptive_initialization(W, extractor)
    current_score = _enhanced_fitness(current_cuts, extractor)
    
    best_cuts = current_cuts.copy()
    best_score = current_score

    tabu_list = collections.deque(maxlen=TABU_MAX)
    tabu_list.append(_hash_cuts(current_cuts))
    
    current_tabu_size = TABU_MIN

    iteration = 0
    stall_count = 0
    last_improvement = 0

    elite_solutions = []
    
    start_time = time.time()
    print(f"[TS++] Start superior，First score: {current_score:.4f}")

    while iteration < MAX_ITERS and time.time() - start_time < TIME_LIMIT:
        iteration += 1

        step = max(1, int(8 * (1 - iteration / MAX_ITERS)))

        neighbors = _generate_neighbors(current_cuts, W, step, extractor)

        neighbors.sort(key=lambda x: x[1], reverse=True)

        next_cuts = None
        next_score = -1
        
        for neighbor, score in neighbors:
            neighbor_hash = _hash_cuts(neighbor)
            
            is_tabu = neighbor_hash in tabu_list

            if (not is_tabu) or (ASPIRATION and score > best_score):
                next_cuts = neighbor
                next_score = score
                break

        if next_cuts is None and neighbors:
            next_cuts, next_score = neighbors[0]

        if next_cuts is not None:
            current_cuts = next_cuts
            current_score = next_score

            tabu_list.append(_hash_cuts(current_cuts))

            if TABU_DYNAMIC:
                if iteration - last_improvement > 30:
                    current_tabu_size = min(TABU_MAX, current_tabu_size + 2)
                else:
                    current_tabu_size = max(TABU_MIN, current_tabu_size - 1)

                while len(tabu_list) > current_tabu_size:
                    tabu_list.popleft()

            if current_score > best_score:
                best_cuts = current_cuts.copy()
                best_score = current_score
                last_improvement = iteration
                stall_count = 0

                elite_solutions.append((best_cuts.copy(), best_score))
                if len(elite_solutions) > 5:
                    elite_solutions = sorted(elite_solutions, key=lambda x: x[1], reverse=True)[:5]
            else:
                stall_count += 1

        if iteration % INTENSIFY_GAP == 0:
            intensified_cuts, intensified_score = _intensification(best_cuts, extractor)
            
            if intensified_score > best_score:
                best_cuts = intensified_cuts
                best_score = intensified_score
                current_cuts = intensified_cuts
                current_score = intensified_score
                last_improvement = iteration
                stall_count = 0

                tabu_list.append(_hash_cuts(current_cuts))

        if iteration % DIVERSIFY_GAP == 0 or stall_count >= STALL_GAP:
            current_cuts = _diversification(best_cuts, W, extractor)
            current_score = _enhanced_fitness(current_cuts, extractor)
            stall_count = 0

            tabu_list.clear()
            tabu_list.append(_hash_cuts(current_cuts))

        if iteration % 20 == 0 or iteration == 1:
            _progress_bar("[TS++]", iteration, MAX_ITERS, 
                        f"score={best_score:.4f} tabu={len(tabu_list)}")

    print("\n[TS++] Final search...")
    
    for i, (elite_cuts, elite_score) in enumerate(elite_solutions):
        refined_cuts, refined_score = _intensification(elite_cuts, extractor)
        
        if refined_score > best_score:
            best_cuts = refined_cuts
            best_score = refined_score
    
    execution_time = time.time() - start_time
    print(f"[TS++] Superior end，use {execution_time:.2f}s，Best score: {best_score:.4f}")
    print(f"[TS++] Best cutting position: {best_cuts}")
    
    return {"cuts": best_cuts} if best_cuts else {}
