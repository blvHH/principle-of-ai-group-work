from __future__ import annotations
import cv2, random, time, sys, math, functools
import numpy as np
from typing import List, Tuple, Dict
import 基础 as base

COLONIES        = 3            
ANTS_PER_COLONY = 25           
ITERATIONS      = 250          
ALPHA           = 1.2          
BETA            = 2.5          
RHO_GLOBAL      = 0.1          
RHO_LOCAL       = 0.3          
Q0              = 0.8          
ELITE_ANTS      = 5            
TIME_LIMIT      = 12.0         
SEED            = 42
random.seed(SEED); np.random.seed(SEED)

class EnhancedFeatureExtractor:
    
    def __init__(self, plate: np.ndarray):
        self.plate = plate
        self.W = plate.shape[1]
        self.H = plate.shape[0]
        
        # 预计算特征
        self.edge_map = self._compute_edge_map()
        self.projection = self._compute_projection()
        self.gradient_map = self._compute_gradient_map()
        self.heuristic_info = self._compute_heuristic_info()
        
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
    
    def _compute_heuristic_info(self) -> np.ndarray:
        projection = self.projection
        heuristic = np.ones(self.W, dtype=float)

        if len(projection) > 0:
            min_val = np.min(projection)
            max_val = np.max(projection)
            
            if max_val > min_val:
                normalized = (max_val - projection) / (max_val - min_val)
                heuristic = normalized + 0.1  
        
        return heuristic

class PheromoneMatrix:
    def __init__(self, W: int):
        self.W = W
        self.matrix = np.ones(W, dtype=float)
        self.tau_min = 0.01
        self.tau_max = 10.0
    
    def get_pheromone(self, position: int) -> float:
        if 0 <= position < self.W:
            return self.matrix[position]
        return self.tau_min
    
    def update_pheromone(self, position: int, delta: float):
        if 0 <= position < self.W:
            self.matrix[position] += delta
            self.matrix[position] = np.clip(self.matrix[position], self.tau_min, self.tau_max)
    
    def evaporate(self, rho: float):
        self.matrix *= (1 - rho)
        self.matrix = np.clip(self.matrix, self.tau_min, self.tau_max)
    
    def local_update(self, position: int, rho_local: float):
        if 0 <= position < self.W:
            self.matrix[position] = (1 - rho_local) * self.matrix[position] + rho_local * self.tau_min

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

def _enhanced_fitness(cuts: List[int], extractor: EnhancedFeatureExtractor) -> float:
    _enhanced_fitness_cached.plate = extractor.plate
    _enhanced_fitness_cached.edge_map = extractor.edge_map
    _enhanced_fitness_cached.projection = extractor.projection
    _enhanced_fitness_cached.gradient_map = extractor.gradient_map
    
    key = ','.join(map(str, cuts))
    return _enhanced_fitness_cached(key, extractor.W, extractor.H)

def _smart_repair(cuts: List[int], W: int, min_gap: int = 4) -> List[int]:
    repaired = cuts.copy()
    
    for i in range(1, 7):
        repaired[i] = max(repaired[i], repaired[i-1] + min_gap)
    
    for i in range(6, 0, -1):
        repaired[i] = min(repaired[i], repaired[i+1] - min_gap)
    
    repaired[0] = 0
    repaired[7] = W
    
    return repaired

class EnhancedAnt:
    def __init__(self, ant_id: int, W: int):
        self.ant_id = ant_id
        self.W = W
        self.path = [0] + [0] * 6 + [W]
        self.fitness = 0.0
        self.visited_positions = set()
    
    def construct_solution(self, pheromone: PheromoneMatrix, 
                          extractor: EnhancedFeatureExtractor,
                          alpha: float, beta: float, q0: float):
        self.path = [0] + [0] * 6 + [self.W]
        self.visited_positions.clear()

        for cut_idx in range(1, 7):
            min_pos = self.path[cut_idx - 1] + 4
            max_pos = self.W - 4 * (7 - cut_idx)
            
            if min_pos >= max_pos:
                self.path[cut_idx] = min_pos
                continue

            candidates = list(range(min_pos, max_pos + 1))
            probabilities = []
            
            for pos in candidates:
                tau = pheromone.get_pheromone(pos)

                eta = extractor.heuristic_info[pos] if pos < len(extractor.heuristic_info) else 0.1

                if pos in self.visited_positions:
                    eta *= 0.5

                prob = (tau ** alpha) * (eta ** beta)
                probabilities.append(prob)

            if random.random() < q0:
                best_idx = np.argmax(probabilities)
                selected_pos = candidates[best_idx]
            else:
                total_prob = sum(probabilities)
                if total_prob > 0:
                    probabilities = [p / total_prob for p in probabilities]
                    selected_pos = np.random.choice(candidates, p=probabilities)
                else:
                    selected_pos = random.choice(candidates)
            
            self.path[cut_idx] = selected_pos
            self.visited_positions.add(selected_pos)

        self.path = _smart_repair(self.path, self.W)

        self.fitness = _enhanced_fitness(self.path, extractor)
    
    def local_search(self, extractor: EnhancedFeatureExtractor):
        improved = True
        while improved:
            improved = False
            
            for idx in range(1, 7):
                for delta in [-2, -1, 1, 2]:
                    candidate = self.path.copy()
                    candidate[idx] += delta

                    if (candidate[idx] > candidate[idx-1] + 4 and 
                        candidate[idx] < candidate[idx+1] - 4):
                        
                        fitness = _enhanced_fitness(candidate, extractor)
                        if fitness > self.fitness + 1e-5:
                            self.path = candidate
                            self.fitness = fitness
                            improved = True

class AntColony:
    def __init__(self, colony_id: int, W: int, extractor: EnhancedFeatureExtractor):
        self.colony_id = colony_id
        self.W = W
        self.extractor = extractor
        self.pheromone = PheromoneMatrix(W)
        self.ants = [EnhancedAnt(i, W) for i in range(ANTS_PER_COLONY)]
        self.best_ant = None
        self.best_fitness = -1.0
        self.elite_ants = []
    
    def iterate(self, alpha: float, beta: float, q0: float):
        for ant in self.ants:
            ant.construct_solution(self.pheromone, self.extractor, alpha, beta, q0)

            for pos in ant.path[1:7]:
                self.pheromone.local_update(pos, RHO_LOCAL)

        self.ants.sort(key=lambda a: a.fitness, reverse=True)
        elite_ants = self.ants[:ELITE_ANTS]
        
        for ant in elite_ants:
            ant.local_search(self.extractor)

        current_best = max(self.ants, key=lambda a: a.fitness)
        if current_best.fitness > self.best_fitness:
            self.best_ant = current_best
            self.best_fitness = current_best.fitness

        self.pheromone.evaporate(RHO_GLOBAL)

        for ant in elite_ants:
            delta_tau = ant.fitness / len(elite_ants)
            for pos in ant.path[1:7]:
                self.pheromone.update_pheromone(pos, delta_tau)

        if self.best_ant:
            delta_tau = self.best_fitness * 2.0
            for pos in self.best_ant.path[1:7]:
                self.pheromone.update_pheromone(pos, delta_tau)

def _multi_colony_cooperation(colonies: List[AntColony]):
    global_best = None
    global_best_fitness = -1.0
    
    for colony in colonies:
        if colony.best_fitness > global_best_fitness:
            global_best = colony.best_ant
            global_best_fitness = colony.best_fitness

    if global_best:
        for colony in colonies:
            delta_tau = global_best_fitness * 0.5
            for pos in global_best.path[1:7]:
                colony.pheromone.update_pheromone(pos, delta_tau)

def _adaptive_parameters(iteration: int, max_iterations: int) -> Tuple[float, float, float]:
    progress = iteration / max_iterations

    alpha = 1.2 + 0.8 * progress
    beta = 2.5 - 1.0 * progress 
    q0 = 0.8 + 0.15 * progress   
    
    return alpha, beta, q0

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
        print("[ACO++] Cannot find license plate ")
        return {}
    
    x, y, w, h = box
    W = w
    plate = base.ensure_plate_binary(img[y:y+h, x:x+w])

    extractor = EnhancedFeatureExtractor(plate)

    colonies = [AntColony(i, W, extractor) for i in range(COLONIES)]
    
    global_best_cuts = None
    global_best_score = -1.0
    start_time = time.time()
    
    print(f"[ACO++] Start superior，{COLONIES} ant colonies，every colony has {ANTS_PER_COLONY} ants")

    for iteration in range(ITERATIONS):
        if time.time() - start_time > TIME_LIMIT:
            break

        alpha, beta, q0 = _adaptive_parameters(iteration, ITERATIONS)

        for colony in colonies:
            colony.iterate(alpha, beta, q0)

        if iteration % 10 == 0:
            _multi_colony_cooperation(colonies)

        for colony in colonies:
            if colony.best_fitness > global_best_score:
                global_best_cuts = colony.best_ant.path.copy()
                global_best_score = colony.best_fitness

        if iteration % 10 == 0 or iteration == 0:
            _progress_bar("[ACO++]", iteration + 1, ITERATIONS, 
                        f"score={global_best_score:.4f}")

    if global_best_cuts:
        print("\n[ACO++] Final search...")

        for scale in [3, 2, 1]:
            improved = True
            while improved:
                improved = False
                
                for idx in range(1, 7):
                    for direction in [-1, 1]:
                        candidate = global_best_cuts.copy()
                        candidate[idx] += direction * scale
                        
                        if (candidate[idx] > candidate[idx-1] + 4 and 
                            candidate[idx] < candidate[idx+1] - 4):
                            
                            score = _enhanced_fitness(candidate, extractor)
                            if score > global_best_score + 1e-5:
                                global_best_cuts = candidate
                                global_best_score = score
                                improved = True
    
    execution_time = time.time() - start_time
    print(f"[ACO++] Superior end，use {execution_time:.2f}s，Best score: {global_best_score:.4f}")
    print(f"[ACO++] Best cutting position: {global_best_cuts}")
    
    return {"cuts": global_best_cuts} if global_best_cuts else {}
