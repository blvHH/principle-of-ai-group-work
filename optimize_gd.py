from __future__ import annotations
import cv2, random, time, sys, math, functools
import numpy as np
from typing import List, Tuple, Dict
import 基础 as base

RESTARTS       = 8           
MAX_ITERS      = 500       
EARLY_WINDOW   = 25          
TIME_LIMIT     = 8.0        
ADAM_B1        = 0.9       
ADAM_B2        = 0.999
ADAM_EPS       = 1e-8
MOMENTUM       = 0.8         
GRAD_CLIP      = 10.0     
SEED           = 42
random.seed(SEED); np.random.seed(SEED)

class EnhancedGradientCalculator:
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

    chars = [plate[:, cuts[i]:cuts[i+1]] for i in range(7)]
    widths = np.array([c.shape[1] for c in chars])

    conf_scores = []
    for char in chars:
        if char.shape[1] > 0:
            conf = base.recognize_one_char_basic(char)[1]
            conf_scores.append(conf)
        else:
            conf_scores.append(0.0)
    avg_confidence = np.mean(conf_scores)

    edge_alignment = 0.0
    for i in range(1, 7):
        if cuts[i] < edge_map.shape[1]:
            edge_strength = np.sum(edge_map[:, max(0, cuts[i]-1):min(edge_map.shape[1], cuts[i]+2)])
            edge_alignment += edge_strength
    edge_alignment = edge_alignment / (6 * H * 3 * 255)

    projection_score = 0.0
    for i in range(1, 7):
        if cuts[i] < len(projection):
            window = projection[max(0, cuts[i]-3):min(len(projection), cuts[i]+4)]
            if len(window) > 0:
                min_idx = np.argmin(window)
                actual_min_pos = max(0, cuts[i]-3) + min_idx
                distance_penalty = abs(cuts[i] - actual_min_pos) / 7.0
                projection_score += max(0, 1.0 - distance_penalty)
    projection_score = projection_score / 6

    ideal_width = W / 7
    width_consistency = 1.0 - np.std(widths) / ideal_width
    width_consistency = max(0, width_consistency)

    completeness_penalty = 0
    for width in widths:
        if width < 6:  
            completeness_penalty += 0.5
        elif width > W * 0.6:  
            completeness_penalty += 0.3
    completeness = max(0, 1.0 - completeness_penalty / 7)

    total_score = (0.45 * avg_confidence + 
                   0.20 * edge_alignment + 
                   0.15 * projection_score + 
                   0.12 * width_consistency + 
                   0.08 * completeness)
    
    return total_score

def _enhanced_fitness(cuts: List[int], calculator: EnhancedGradientCalculator) -> float:
    _enhanced_fitness_cached.plate = calculator.plate
    _enhanced_fitness_cached.edge_map = calculator.edge_map
    _enhanced_fitness_cached.projection = calculator.projection
    _enhanced_fitness_cached.gradient_map = calculator.gradient_map
    
    key = ','.join(map(str, cuts))
    return _enhanced_fitness_cached(key, calculator.W, calculator.H)

def _smart_repair(cuts: List[int], W: int, min_gap: int = 4) -> List[int]:
    repaired = cuts.copy()

    for i in range(1, 7):
        repaired[i] = max(repaired[i], repaired[i-1] + min_gap)
    
    for i in range(6, 0, -1):
        repaired[i] = min(repaired[i], repaired[i+1] - min_gap)

    repaired[0] = 0
    repaired[7] = W
    
    return repaired

def _numerical_gradient(cuts: List[int], calculator: EnhancedGradientCalculator, h: int = 1) -> np.ndarray:
    base_score = _enhanced_fitness(cuts, calculator)
    grad = np.zeros(6, dtype=float)
    
    for i in range(1, 7):
        if cuts[i] - h <= cuts[i-1] or cuts[i] + h >= cuts[i+1]:
            continue

        cuts_left = cuts.copy()
        cuts_right = cuts.copy()
        
        cuts_left[i] -= h
        cuts_right[i] += h
        
        score_left = _enhanced_fitness(cuts_left, calculator)
        score_right = _enhanced_fitness(cuts_right, calculator)

        grad[i-1] = (score_right - score_left) / (2 * h)
    
    return grad

def _adam_update(cuts: List[int], calculator: EnhancedGradientCalculator, 
                m: np.ndarray, v: np.ndarray, t: int, lr: float) -> Tuple[List[int], np.ndarray, np.ndarray]:
    grad = _numerical_gradient(cuts, calculator)

    grad_norm = np.linalg.norm(grad)
    if grad_norm > GRAD_CLIP:
        grad = grad * GRAD_CLIP / grad_norm

    m = ADAM_B1 * m + (1 - ADAM_B1) * grad
    v = ADAM_B2 * v + (1 - ADAM_B2) * (grad ** 2)
    
    m_hat = m / (1 - ADAM_B1 ** t)
    v_hat = v / (1 - ADAM_B2 ** t)

    delta = lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

    new_cuts = cuts.copy()
    for i in range(1, 7):
        new_cuts[i] = int(cuts[i] + delta[i-1])

    new_cuts = _smart_repair(new_cuts, calculator.W)
    
    return new_cuts, m, v

def _momentum_update(cuts: List[int], calculator: EnhancedGradientCalculator, 
                    velocity: np.ndarray, lr: float) -> Tuple[List[int], np.ndarray]:
    grad = _numerical_gradient(cuts, calculator)

    grad_norm = np.linalg.norm(grad)
    if grad_norm > GRAD_CLIP:
        grad = grad * GRAD_CLIP / grad_norm

    velocity = MOMENTUM * velocity + lr * grad

    new_cuts = cuts.copy()
    for i in range(1, 7):
        new_cuts[i] = int(cuts[i] + velocity[i-1])

    new_cuts = _smart_repair(new_cuts, calculator.W)
    
    return new_cuts, velocity

def _line_search(cuts: List[int], grad: np.ndarray, calculator: EnhancedGradientCalculator) -> List[int]:
    base_score = _enhanced_fitness(cuts, calculator)
    best_cuts = cuts.copy()
    best_score = base_score

    if np.linalg.norm(grad) > 0:
        direction = grad / np.linalg.norm(grad)
    else:
        return cuts

    for step_size in [0.5, 1, 2, 4, 8, 16]:
        candidate = cuts.copy()
        for i in range(1, 7):
            candidate[i] = int(cuts[i] + step_size * direction[i-1])
        
        candidate = _smart_repair(candidate, calculator.W)
        score = _enhanced_fitness(candidate, calculator)
        
        if score > best_score:
            best_cuts = candidate
            best_score = score
    
    return best_cuts

def _adaptive_initialization(W: int, calculator: EnhancedGradientCalculator) -> List[List[int]]:
    initializations = []

    uniform_cuts = [0] + [int(i * W / 7) for i in range(1, 7)] + [W]
    initializations.append(uniform_cuts)

    projection = calculator.projection
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

    edge_map = calculator.edge_map
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

    for _ in range(RESTARTS - len(initializations)):
        random_cuts = [0] + sorted([
            int(i * W / 7 + random.uniform(-0.25, 0.25) * W / 7) 
            for i in range(1, 7)
        ]) + [W]
        random_cuts = _smart_repair(random_cuts, calculator.W)
        initializations.append(random_cuts)
    
    return initializations

def _multi_scale_local_search(cuts: List[int], calculator: EnhancedGradientCalculator) -> Tuple[List[int], float]:
    best_cuts = cuts.copy()
    best_score = _enhanced_fitness(best_cuts, calculator)

    scales = [3, 2, 1]
    
    for scale in scales:
        improved = True
        while improved:
            improved = False

            for idx in range(1, 7):
                for direction in [-1, 1]:
                    candidate = best_cuts.copy()
                    candidate[idx] += direction * scale

                    if (candidate[idx] > candidate[idx-1] + 4 and 
                        candidate[idx] < candidate[idx+1] - 4):
                        
                        score = _enhanced_fitness(candidate, calculator)
                        if score > best_score + 1e-5:
                            best_cuts, best_score = candidate, score
                            improved = True
    
    return best_cuts, best_score

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
        print("[GD++] Cannot find license plate")
        return {}
    
    x, y, w, h = box
    W = w
    plate = base.ensure_plate_binary(img[y:y+h, x:x+w])

    calculator = EnhancedGradientCalculator(plate)

    initializations = _adaptive_initialization(W, calculator)
    
    global_best_cuts = None
    global_best_score = -1.0
    start_time = time.time()
    
    print(f"[GD++] Start superior，has {len(initializations)} starting point")
    
    for restart_idx, initial_cuts in enumerate(initializations):
        if time.time() - start_time > TIME_LIMIT:
            break

        current_cuts = _smart_repair(initial_cuts, calculator.W)
        current_score = _enhanced_fitness(current_cuts, calculator)
        
        best_cuts = current_cuts.copy()
        best_score = current_score

        m = np.zeros(6) 
        v = np.zeros(6)
        velocity = np.zeros(6)

        base_lr = 5.0
        stall_count = 0

        for it in range(1, MAX_ITERS + 1):
            lr = base_lr * (1.0 - 0.8 * it / MAX_ITERS)

            if it % 3 == 0:
                new_cuts, velocity = _momentum_update(current_cuts, calculator, velocity, lr)
            else:
                new_cuts, m, v = _adam_update(current_cuts, calculator, m, v, it, lr)

            if it % 20 == 0:
                grad = _numerical_gradient(current_cuts, calculator)
                new_cuts = _line_search(current_cuts, grad, calculator)
            
            new_score = _enhanced_fitness(new_cuts, calculator)

            if new_score > current_score:
                current_cuts = new_cuts
                current_score = new_score

                if current_score > best_score:
                    best_cuts = current_cuts.copy()
                    best_score = current_score
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                stall_count += 1

            if stall_count >= EARLY_WINDOW:
                break

            if it % 20 == 0 or it == 1:
                _progress_bar(f"[GD++#{restart_idx+1}]", it, MAX_ITERS, f"score={best_score:.4f}")

        best_cuts, best_score = _multi_scale_local_search(best_cuts, calculator)
        
        print(f"\n[GD++#{restart_idx+1}] Best score: {best_score:.4f}")

        if best_score > global_best_score:
            global_best_cuts = best_cuts.copy()
            global_best_score = best_score
    
    execution_time = time.time() - start_time
    print(f"[GD++] Superior end，use {execution_time:.2f}s，Best score: {global_best_score:.4f}")
    print(f"[GD++] Best cutting position: {global_best_cuts}")
    
    return {"cuts": global_best_cuts} if global_best_cuts else {}
