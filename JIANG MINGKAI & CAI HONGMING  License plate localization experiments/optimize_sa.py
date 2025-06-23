from __future__ import annotations
import cv2, random, time, sys, math, functools
import numpy as np
from typing import List, Tuple, Dict
import 基础 as base

CHAINS          = 4            
INITIAL_TEMP    = 100.0       
FINAL_TEMP      = 0.01         
COOLING_RATES   = [0.95, 0.96, 0.97, 0.98]  
MAX_ITERATIONS  = 400          
REHEAT_FREQ     = 80           
REHEAT_FACTOR   = 2.0         
ELITE_MEMORY    = 10           
RESTART_FREQ    = 150          
TIME_LIMIT      = 12.0        
SEED            = 999
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
    
    # 5. 字符完整性 (权重: 0.08)
    completeness_penalty = 0
    for width in widths:
        if width < 6:
            completeness_penalty += 0.5
        elif width > W * 0.6:
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
    
    for i in range(1, 7):
        repaired[i] = max(repaired[i], repaired[i-1] + min_gap)
    
    for i in range(6, 0, -1):
        repaired[i] = min(repaired[i], repaired[i+1] - min_gap)
    
    repaired[0] = 0
    repaired[7] = W
    
    return repaired

def _generate_neighbor_single(cuts: List[int], W: int, step_size: int) -> List[int]:
    neighbor = cuts.copy()
    idx = random.randint(1, 6)
    delta = random.randint(-step_size, step_size)
    neighbor[idx] += delta
    return _smart_repair(neighbor, W)

def _generate_neighbor_multi(cuts: List[int], W: int, step_size: int) -> List[int]:
    neighbor = cuts.copy()
    num_changes = random.randint(1, 3)
    indices = random.sample(range(1, 7), num_changes)
    
    for idx in indices:
        delta = random.randint(-step_size, step_size)
        neighbor[idx] += delta
    
    return _smart_repair(neighbor, W)

def _generate_neighbor_projection(cuts: List[int], extractor: EnhancedFeatureExtractor, step_size: int) -> List[int]:
    neighbor = cuts.copy()
    projection = extractor.projection

    indices = random.sample(range(1, 7), random.randint(1, 2))
    
    for idx in indices:
        current_pos = neighbor[idx]

        search_range = min(step_size * 2, 20)
        start = max(neighbor[idx-1] + 4, current_pos - search_range)
        end = min(neighbor[idx+1] - 4, current_pos + search_range)
        
        if start < end and end < len(projection):
            window = projection[start:end+1]
            if len(window) > 0:
                min_idx = np.argmin(window)
                neighbor[idx] = start + min_idx
    
    return _smart_repair(neighbor, extractor.W)

def _generate_neighbor_adaptive(cuts: List[int], extractor: EnhancedFeatureExtractor, 
                               step_size: int, temperature: float) -> List[int]:
    if temperature > 50:
        strategy = random.random()
        if strategy < 0.5:
            return _generate_neighbor_multi(cuts, extractor.W, step_size)
        else:
            return _generate_neighbor_single(cuts, extractor.W, step_size)
    else:
        strategy = random.random()
        if strategy < 0.6:
            return _generate_neighbor_projection(cuts, extractor, step_size)
        else:
            return _generate_neighbor_single(cuts, extractor.W, step_size)

class EliteMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.solutions = []
    
    def add(self, cuts: List[int], fitness: float):
        for i, (stored_cuts, stored_fitness) in enumerate(self.solutions):
            if stored_cuts == cuts:
                if fitness > stored_fitness:
                    self.solutions[i] = (cuts.copy(), fitness)
                return

        self.solutions.append((cuts.copy(), fitness))

        if len(self.solutions) > self.capacity:
            self.solutions.sort(key=lambda x: x[1], reverse=True)
            self.solutions = self.solutions[:self.capacity]
    
    def get_best(self) -> Tuple[List[int], float]:
        if self.solutions:
            return max(self.solutions, key=lambda x: x[1])
        return None, -1.0
    
    def get_random_elite(self) -> Tuple[List[int], float]:
        if self.solutions:
            return random.choice(self.solutions)
        return None, -1.0

class AnnealingChain:
    def __init__(self, chain_id: int, W: int, extractor: EnhancedFeatureExtractor, 
                 cooling_rate: float, elite_memory: EliteMemory):
        self.chain_id = chain_id
        self.W = W
        self.extractor = extractor
        self.cooling_rate = cooling_rate
        self.elite_memory = elite_memory

        self.current_cuts = self._initialize_solution()
        self.current_fitness = _enhanced_fitness(self.current_cuts, extractor)
        
        self.best_cuts = self.current_cuts.copy()
        self.best_fitness = self.current_fitness

        self.temperature = INITIAL_TEMP
        self.iteration = 0
        self.accepted_moves = 0
        self.rejected_moves = 0

        self.elite_memory.add(self.best_cuts, self.best_fitness)
    
    def _initialize_solution(self) -> List[int]:
        if self.chain_id == 0:
            return [0] + [int(i * self.W / 7) for i in range(1, 7)] + [self.W]
        elif self.chain_id == 1:
            projection = self.extractor.projection
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
                        score = depth
                        valley_scores.append((v, score))
                    
                    valley_scores.sort(key=lambda x: x[1], reverse=True)
                    selected_valleys = sorted([v[0] for v in valley_scores[:6]])
                    
                    return [0] + selected_valleys + [self.W]

            return [0] + sorted([
                random.randint(int(i*self.W/7*0.8), int(i*self.W/7*1.2)) 
                for i in range(1, 7)
            ]) + [self.W]
        else:
            return [0] + sorted([
                random.randint(int(i*self.W/7*0.7), int(i*self.W/7*1.3)) 
                for i in range(1, 7)
            ]) + [self.W]
    
    def anneal_step(self):
        self.iteration += 1
        
        if self.temperature > 50:
            step_size = max(8, int(self.W * 0.08))
        elif self.temperature > 10:
            step_size = max(4, int(self.W * 0.04))
        else:
            step_size = max(2, int(self.W * 0.02))

        neighbor = _generate_neighbor_adaptive(
            self.current_cuts, self.extractor, step_size, self.temperature
        )
        neighbor_fitness = _enhanced_fitness(neighbor, self.extractor)

        delta = neighbor_fitness - self.current_fitness
        
        if delta > 0:
            self.current_cuts = neighbor
            self.current_fitness = neighbor_fitness
            self.accepted_moves += 1

            if neighbor_fitness > self.best_fitness:
                self.best_cuts = neighbor.copy()
                self.best_fitness = neighbor_fitness

                self.elite_memory.add(self.best_cuts, self.best_fitness)
        
        elif self.temperature > FINAL_TEMP:
            probability = math.exp(min(700, delta / self.temperature))
            if random.random() < probability:
                self.current_cuts = neighbor
                self.current_fitness = neighbor_fitness
                self.accepted_moves += 1
            else:
                self.rejected_moves += 1
        else:
            self.rejected_moves += 1

        self.temperature *= self.cooling_rate

        if (self.iteration % REHEAT_FREQ == 0 and 
            self.accepted_moves < self.rejected_moves * 0.1):
            self.temperature = min(INITIAL_TEMP, self.temperature * REHEAT_FACTOR)
    
    def restart(self):
        elite_cuts, elite_fitness = self.elite_memory.get_random_elite()
        
        if elite_cuts is not None:
            noise_cuts = elite_cuts.copy()
            for i in range(1, 7):
                noise = random.randint(-int(self.W * 0.1), int(self.W * 0.1))
                noise_cuts[i] += noise
            
            self.current_cuts = _smart_repair(noise_cuts, self.W)
        else:
            self.current_cuts = self._initialize_solution()
        
        self.current_fitness = _enhanced_fitness(self.current_cuts, self.extractor)
        self.temperature = INITIAL_TEMP * 0.5
        self.accepted_moves = 0
        self.rejected_moves = 0
    
    def local_search(self):
        best_cuts = self.best_cuts.copy()
        best_score = self.best_fitness

        for scale in [2, 1]:
            improved = True
            while improved:
                improved = False
                
                for idx in range(1, 7):
                    for direction in [-1, 1]:
                        candidate = best_cuts.copy()
                        candidate[idx] += direction * scale
                        
                        if (candidate[idx] > candidate[idx-1] + 4 and 
                            candidate[idx] < candidate[idx+1] - 4):
                            
                            score = _enhanced_fitness(candidate, self.extractor)
                            if score > best_score + 1e-5:
                                best_cuts = candidate
                                best_score = score
                                improved = True

        if best_score > self.best_fitness:
            self.best_cuts = best_cuts
            self.best_fitness = best_score
            self.elite_memory.add(self.best_cuts, self.best_fitness)

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
        print("[SA++] Cannot find license plate")
        return {}
    
    x, y, w, h = box
    W = w
    plate = base.ensure_plate_binary(img[y:y+h, x:x+w])

    extractor = EnhancedFeatureExtractor(plate)

    elite_memory = EliteMemory(ELITE_MEMORY)

    chains = [AnnealingChain(i, W, extractor, COOLING_RATES[i % len(COOLING_RATES)], elite_memory) 
              for i in range(CHAINS)]
    
    global_best_cuts = None
    global_best_score = -1.0
    start_time = time.time()
    
    print(f"[SA++] Start superior，has {CHAINS} chains")

    for iteration in range(MAX_ITERATIONS):
        if time.time() - start_time > TIME_LIMIT:
            break

        for chain in chains:
            chain.anneal_step()

        if iteration % RESTART_FREQ == 0 and iteration > 0:
            chains.sort(key=lambda c: c.best_fitness)
            worst_chain = chains[0]
            worst_chain.restart()

        if iteration % 50 == 0:
            for chain in chains:
                chain.local_search()

        for chain in chains:
            if chain.best_fitness > global_best_score:
                global_best_cuts = chain.best_cuts.copy()
                global_best_score = chain.best_fitness

        if iteration % 20 == 0 or iteration == 0:
            avg_temp = np.mean([chain.temperature for chain in chains])
            _progress_bar("[SA++]", iteration + 1, MAX_ITERATIONS, 
                        f"score={global_best_score:.4f} temp={avg_temp:.2f}")

    if global_best_cuts:
        print("\n[SA++] Final search...")

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
    print(f"[SA++] Superior end，use {execution_time:.2f}s，Best score: {global_best_score:.4f}")
    print(f"[SA++] Best cutting position: {global_best_cuts}")
    
    return {"cuts": global_best_cuts} if global_best_cuts else {}
