from __future__ import annotations
import cv2, random, time, sys, math, functools
import numpy as np
from typing import List, Tuple, Dict
import 基础 as base

SWARMS          = 4            
PARTICLES       = 30           
ITERATIONS      = 350          
W_MIN, W_MAX    = 0.4, 0.9    
C1_MIN, C1_MAX  = 1.5, 2.5    
C2_MIN, C2_MAX  = 1.5, 2.5  
V_MAX_FACTOR    = 0.3        
MUTATION_RATE   = 0.15         
ELITE_COUNT     = 3            
EXCHANGE_FREQ   = 25          
TIME_LIMIT      = 12.0         
SEED            = 333
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
        
        # 垂直边缘增强
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
            edge_strength = np.sum(edge_map[:, max(0, cuts[i]-1):min(edge_map.shape[1], cuts[i]+2)])
            edge_alignment += edge_strength
    edge_alignment = edge_alignment / (6 * H * 3 * 255)
    
    # 3. 投影谷值对齐 (权重: 0.15)
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
    
    # 4. 宽度一致性 (权重: 0.12)
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

class EnhancedParticle:
    def __init__(self, particle_id: int, W: int):
        self.particle_id = particle_id
        self.W = W

        self.position = np.array([0] + sorted([
            random.randint(int(i*W/7*0.7), int(i*W/7*1.3)) 
            for i in range(1, 7)
        ]) + [W], dtype=float)

        v_max = W * V_MAX_FACTOR
        self.velocity = np.random.uniform(-v_max, v_max, size=8)
        self.velocity[0] = 0  
        self.velocity[7] = 0 

        self.best_position = self.position.copy()
        self.best_fitness = -1.0

        self.fitness = -1.0

        self.stagnation_count = 0
    
    def update_velocity(self, global_best_position: np.ndarray, 
                       w: float, c1: float, c2: float):
        r1 = np.random.random(8)
        r2 = np.random.random(8)

        self.velocity = (w * self.velocity + 
                        c1 * r1 * (self.best_position - self.position) + 
                        c2 * r2 * (global_best_position - self.position))

        v_max = self.W * V_MAX_FACTOR * (1.0 - self.stagnation_count * 0.01)
        self.velocity = np.clip(self.velocity, -v_max, v_max)

        self.velocity[0] = 0
        self.velocity[7] = 0
    
    def update_position(self):
        self.position += self.velocity

        cuts_int = [int(pos) for pos in self.position]
        cuts_repaired = _smart_repair(cuts_int, self.W)
        self.position = np.array(cuts_repaired, dtype=float)
    
    def evaluate(self, extractor: EnhancedFeatureExtractor):
        cuts_int = [int(pos) for pos in self.position]
        self.fitness = _enhanced_fitness(cuts_int, extractor)

        if self.fitness > self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = self.fitness
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
    
    def mutate(self, extractor: EnhancedFeatureExtractor):
        if random.random() < MUTATION_RATE:
            mutation_points = random.sample(range(1, 7), random.randint(1, 2))
            
            for idx in mutation_points:
                projection = extractor.projection
                current_pos = int(self.position[idx])

                search_range = min(20, self.W // 10)
                start = max(int(self.position[idx-1]) + 4, current_pos - search_range)
                end = min(int(self.position[idx+1]) - 4, current_pos + search_range)
                
                if start < end and end < len(projection):
                    window = projection[start:end+1]
                    if len(window) > 0:
                        min_idx = np.argmin(window)
                        new_pos = start + min_idx
                        self.position[idx] = float(new_pos)
    
    def local_search(self, extractor: EnhancedFeatureExtractor):
        current_cuts = [int(pos) for pos in self.position]
        best_cuts = current_cuts.copy()
        best_score = _enhanced_fitness(best_cuts, extractor)

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
                            
                            score = _enhanced_fitness(candidate, extractor)
                            if score > best_score + 1e-5:
                                best_cuts = candidate
                                best_score = score
                                improved = True

        if best_score > self.fitness:
            self.position = np.array(best_cuts, dtype=float)
            self.fitness = best_score
            
            if best_score > self.best_fitness:
                self.best_position = self.position.copy()
                self.best_fitness = best_score
                self.stagnation_count = 0

class ParticleSwarm:
    def __init__(self, swarm_id: int, W: int, extractor: EnhancedFeatureExtractor):
        self.swarm_id = swarm_id
        self.W = W
        self.extractor = extractor
        self.particles = [EnhancedParticle(i, W) for i in range(PARTICLES)]

        self.global_best_position = None
        self.global_best_fitness = -1.0

        for particle in self.particles:
            particle.evaluate(extractor)
            
            if particle.fitness > self.global_best_fitness:
                self.global_best_position = particle.position.copy()
                self.global_best_fitness = particle.fitness
    
    def iterate(self, iteration: int, max_iterations: int):
        progress = iteration / max_iterations
        w = W_MAX - (W_MAX - W_MIN) * progress  
        c1 = C1_MAX - (C1_MAX - C1_MIN) * progress 
        c2 = C2_MIN + (C2_MAX - C2_MIN) * progress 

        for particle in self.particles:
            particle.update_velocity(self.global_best_position, w, c1, c2)
            particle.update_position()

            particle.evaluate(self.extractor)

            particle.mutate(self.extractor)

            if particle.fitness > self.global_best_fitness:
                self.global_best_position = particle.position.copy()
                self.global_best_fitness = particle.fitness

        if iteration % 20 == 0:
            elite_particles = sorted(self.particles, key=lambda p: p.fitness, reverse=True)[:ELITE_COUNT]
            for particle in elite_particles:
                particle.local_search(self.extractor)
                
                if particle.fitness > self.global_best_fitness:
                    self.global_best_position = particle.position.copy()
                    self.global_best_fitness = particle.fitness

def _swarm_cooperation(swarms: List[ParticleSwarm]):
    global_best = None
    global_best_fitness = -1.0
    
    for swarm in swarms:
        if swarm.global_best_fitness > global_best_fitness:
            global_best = swarm.global_best_position.copy()
            global_best_fitness = swarm.global_best_fitness

    if global_best is not None:
        swarms.sort(key=lambda s: s.global_best_fitness)

        for i in range(len(swarms) // 2):
            swarm = swarms[i]

            num_replace = PARTICLES // 4
            replace_indices = random.sample(range(PARTICLES), num_replace)
            
            for idx in replace_indices:
                noise = np.random.normal(0, swarm.W * 0.05, size=8)
                noise[0] = 0  
                noise[7] = 0
                
                new_position = global_best + noise
                cuts_int = [int(pos) for pos in new_position]
                cuts_repaired = _smart_repair(cuts_int, swarm.W)
                
                swarm.particles[idx].position = np.array(cuts_repaired, dtype=float)
                swarm.particles[idx].velocity *= 0.5 

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
        print("[PSO++] Cannot find license plate")
        return {}
    
    x, y, w, h = box
    W = w
    plate = base.ensure_plate_binary(img[y:y+h, x:x+w])

    extractor = EnhancedFeatureExtractor(plate)

    swarms = [ParticleSwarm(i, W, extractor) for i in range(SWARMS)]
    
    global_best_cuts = None
    global_best_score = -1.0
    start_time = time.time()
    
    print(f"[PSO++] Start superior，{SWARMS} swarms，every swarm has {PARTICLES} particle")

    for iteration in range(ITERATIONS):
        if time.time() - start_time > TIME_LIMIT:
            break

        for swarm in swarms:
            swarm.iterate(iteration, ITERATIONS)

        if iteration % EXCHANGE_FREQ == 0:
            _swarm_cooperation(swarms)

        for swarm in swarms:
            if swarm.global_best_fitness > global_best_score:
                global_best_cuts = [int(pos) for pos in swarm.global_best_position]
                global_best_score = swarm.global_best_fitness

        if iteration % 15 == 0 or iteration == 0:
            _progress_bar("[PSO++]", iteration + 1, ITERATIONS, 
                        f"score={global_best_score:.4f}")

    if global_best_cuts:
        print("\n[PSO++] Final search...")

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
    print(f"[PSO++] Superior end，use {execution_time:.2f}s，Best score: {global_best_score:.4f}")
    print(f"[PSO++] Best cutting position: {global_best_cuts}")
    
    return {"cuts": global_best_cuts} if global_best_cuts else {}
