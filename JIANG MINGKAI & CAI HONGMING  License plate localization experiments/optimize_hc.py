from __future__ import annotations
import cv2, random, time, sys, math, functools
import numpy as np
from typing import List, Tuple, Dict
import 基础 as base

N_RESTART        = 20     
ITER_COARSE      = 200    
ITER_FINE        = 120    
ITER_MICRO       = 60       
TIME_LIMIT       = 10.0   
TEMP0            = 0.25     
DECAY_ADAPTIVE   = True    
WIN_STALL        = 35      
EPS_IMPROVE      = 5e-5   
SEED             = 42
random.seed(SEED); np.random.seed(SEED)

class EnhancedFitnessCalculator:

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
        edges3 = cv2.Canny(self.plate, 80, 200)

        combined = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))

        kernel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        vertical_edges = cv2.filter2D(self.plate.astype(np.float32), -1, kernel_v)
        vertical_edges = np.abs(vertical_edges).astype(np.uint8)
        
        return cv2.bitwise_or(combined, vertical_edges)
    
    def _compute_projection(self) -> np.ndarray:
        binary = (self.plate < 128).astype(np.uint8)
        return np.sum(binary, axis=0)
    
    def _compute_gradient_map(self) -> np.ndarray:
        grad_x = cv2.Sobel(self.plate, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.plate, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(grad_x**2 + grad_y**2)

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

class EnhancedFitnessCache:
    plate: np.ndarray = None
    edge_map: np.ndarray = None
    projection: np.ndarray = None
    gradient_map: np.ndarray = None

def _enhanced_fitness(cuts: List[int], calculator: EnhancedFitnessCalculator) -> float:
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
    repaired[6] = min(repaired[6], W - min_gap)
    
    return repaired

def _intelligent_neighbor(cuts: List[int], W: int, step: int, iteration: int, 
                         calculator: EnhancedFitnessCalculator) -> List[int]:
    neighbor = cuts.copy()

    strategy = random.random()
    
    if strategy < 0.3:
        _edge_guided_move(neighbor, calculator, step)
    elif strategy < 0.6:
        _projection_guided_move(neighbor, calculator, step)
    elif strategy < 0.8:
        _symmetric_adjustment(neighbor, W, step)
    else: 
        _random_perturbation(neighbor, W, step)
    
    return _smart_repair(neighbor, W)

def _edge_guided_move(cuts: List[int], calculator: EnhancedFitnessCalculator, step: int):
    edge_map = calculator.edge_map

    num_moves = random.randint(1, 3)
    indices = random.sample(range(1, 7), num_moves)
    
    for idx in indices:
        current_pos = cuts[idx]

        search_range = min(step * 2, 15)
        start = max(cuts[idx-1] + 4, current_pos - search_range)
        end = min(cuts[idx+1] - 4, current_pos + search_range)
        
        if start < end:
            edge_strengths = []
            positions = []
            
            for pos in range(start, end + 1):
                if pos < edge_map.shape[1]:
                    strength = np.sum(edge_map[:, max(0, pos-1):min(edge_map.shape[1], pos+2)])
                    edge_strengths.append(strength)
                    positions.append(pos)
            
            if edge_strengths:
                best_idx = np.argmax(edge_strengths)
                cuts[idx] = positions[best_idx]

def _projection_guided_move(cuts: List[int], calculator: EnhancedFitnessCalculator, step: int):
    projection = calculator.projection

    num_moves = random.randint(1, 2)
    indices = random.sample(range(1, 7), num_moves)
    
    for idx in indices:
        current_pos = cuts[idx]

        search_range = min(step * 3, 20)
        start = max(cuts[idx-1] + 4, current_pos - search_range)
        end = min(cuts[idx+1] - 4, current_pos + search_range)
        
        if start < end and end < len(projection):
            window = projection[start:end+1]
            if len(window) > 0:
                min_idx = np.argmin(window)
                cuts[idx] = start + min_idx

def _symmetric_adjustment(cuts: List[int], W: int, step: int):
    if random.random() < 0.5:
        center = W // 2
        factor = 1.0 + random.uniform(-0.1, 0.1)
        
        for i in range(1, 7):
            offset = cuts[i] - center
            cuts[i] = int(center + offset * factor)
    else:
        delta = random.randint(-step, step)
        for i in range(1, 4):
            cuts[i] += delta
            cuts[7-i] -= delta

def _random_perturbation(cuts: List[int], W: int, step: int):
    num_moves = random.randint(1, 4)
    indices = random.sample(range(1, 7), num_moves)
    
    for idx in indices:
        delta = random.randint(-step, step)
        cuts[idx] += delta

def _multi_scale_local_search(cuts: List[int], calculator: EnhancedFitnessCalculator) -> Tuple[List[int], float]:
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
                        if score > best_score + EPS_IMPROVE:
                            best_cuts, best_score = candidate, score
                            improved = True
    
    return best_cuts, best_score

def _adaptive_initialization(W: int, calculator: EnhancedFitnessCalculator) -> List[List[int]]:
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

    for _ in range(N_RESTART - len(initializations)):
        random_cuts = [0] + sorted([
            int(i * W / 7 + random.uniform(-0.25, 0.25) * W / 7) 
            for i in range(1, 7)
        ]) + [W]
        random_cuts = _smart_repair(random_cuts, W)
        initializations.append(random_cuts)
    
    return initializations

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
        print("[HC++] Cannot find license plate")
        return {}
    
    x, y, w, h = box
    W = w
    plate = base.ensure_plate_binary(img[y:y+h, x:x+w])

    calculator = EnhancedFitnessCalculator(plate)

    initializations = _adaptive_initialization(W, calculator)
    
    global_best_cuts = None
    global_best_score = -1.0
    start_time = time.time()
    
    print(f"[HC++] Start superior，has {len(initializations)} starting point")
    
    for restart_idx, initial_cuts in enumerate(initializations):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        current_cuts = _smart_repair(initial_cuts, W)
        current_score = _enhanced_fitness(current_cuts, calculator)
        
        best_cuts = current_cuts.copy()
        best_score = current_score

        temperature = TEMP0
        stall_count = 0
        last_improvement = current_score
        
        total_iterations = ITER_COARSE + ITER_FINE + ITER_MICRO
        iteration_count = 0

        coarse_step = max(8, int(W * 0.08))
        for iter_coarse in range(ITER_COARSE):
            iteration_count += 1

            progress = iter_coarse / ITER_COARSE
            adaptive_step = int(coarse_step * (1 - progress * 0.6))

            neighbor = _intelligent_neighbor(current_cuts, W, adaptive_step, 
                                           iteration_count, calculator)
            neighbor_score = _enhanced_fitness(neighbor, calculator)

            delta = neighbor_score - current_score
            if delta > 0 or (temperature > 0 and 
                           math.exp(min(700, delta / temperature)) > random.random()):
                current_cuts = neighbor
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_cuts = current_cuts.copy()
                    best_score = current_score
                    stall_count = 0
                    last_improvement = current_score
                else:
                    stall_count += 1
            else:
                stall_count += 1

            if DECAY_ADAPTIVE:
                if stall_count > 10:
                    temperature *= 0.95
                else:
                    temperature *= 0.98
            else:
                temperature *= 0.97

            if stall_count >= WIN_STALL:
                break

            if iteration_count % 20 == 0:
                _progress_bar(f"[HC++#{restart_idx+1}]", iteration_count, total_iterations, 
                            f"score={best_score:.4f}")

        current_cuts = best_cuts.copy()
        current_score = best_score
        temperature = TEMP0 * 0.3
        fine_step = max(3, int(W * 0.03))
        
        for iter_fine in range(ITER_FINE):
            iteration_count += 1
            
            progress = iter_fine / ITER_FINE
            adaptive_step = max(1, int(fine_step * (1 - progress * 0.8)))
            
            neighbor = _intelligent_neighbor(current_cuts, W, adaptive_step, 
                                           iteration_count, calculator)
            neighbor_score = _enhanced_fitness(neighbor, calculator)
            
            delta = neighbor_score - current_score
            if delta > 0 or (temperature > 0 and 
                           math.exp(min(700, delta / temperature)) > random.random()):
                current_cuts = neighbor
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_cuts = current_cuts.copy()
                    best_score = current_score
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                stall_count += 1
            
            temperature *= 0.96
            
            if stall_count >= WIN_STALL // 2:
                break
            
            if iteration_count % 15 == 0:
                _progress_bar(f"[HC++#{restart_idx+1}]", iteration_count, total_iterations, 
                            f"score={best_score:.4f}")

        best_cuts, best_score = _multi_scale_local_search(best_cuts, calculator)
        iteration_count += ITER_MICRO
        
        _progress_bar(f"[HC++#{restart_idx+1}]", iteration_count, total_iterations, 
                    f"final={best_score:.4f}")

        if best_score > global_best_score:
            global_best_cuts = best_cuts.copy()
            global_best_score = best_score
        
        print()
    
    execution_time = time.time() - start_time
    print(f"[HC++] Superior end，use {execution_time:.2f}s，Best score: {global_best_score:.4f}")
    print(f"[HC++] Best cutting position: {global_best_cuts}")
    
    return {"cuts": global_best_cuts} if global_best_cuts else {}
