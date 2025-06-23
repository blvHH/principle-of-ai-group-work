from __future__ import annotations
import cv2, random, time, sys, math, functools
import numpy as np
from typing import List, Tuple, Dict
import 基础 as base

POPULATIONS     = 3            
POP_SIZE        = 50          
GENERATIONS     = 300        
ELITE_RATE      = 0.15        
CROSSOVER_RATE  = 0.85      
MUTATION_RATE   = 0.25       
TOURNAMENT_SIZE = 5           
MIGRATION_FREQ  = 20         
MIGRATION_RATE  = 0.1    
LOCAL_SEARCH_FREQ = 30        
TIME_LIMIT      = 12.0     
SEED            = 777
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

class Individual:

    def __init__(self, W: int, cuts: List[int] = None):
        self.W = W
        if cuts is None:
            self.cuts = [0] + sorted([
                random.randint(int(i*W/7*0.7), int(i*W/7*1.3)) 
                for i in range(1, 7)
            ]) + [W]
            self.cuts = _smart_repair(self.cuts, W)
        else:
            self.cuts = cuts.copy()
        
        self.fitness = -1.0
        self.age = 0
    
    def evaluate(self, extractor: EnhancedFeatureExtractor):
        self.fitness = _enhanced_fitness(self.cuts, extractor)
    
    def local_search(self, extractor: EnhancedFeatureExtractor):
        best_cuts = self.cuts.copy()
        best_score = self.fitness

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
            self.cuts = best_cuts
            self.fitness = best_score

def _tournament_selection(population: List[Individual], tournament_size: int) -> Individual:
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda ind: ind.fitness)

def _single_point_crossover(parent1: Individual, parent2: Individual, W: int) -> Tuple[Individual, Individual]:
    crossover_point = random.randint(1, 6)
    
    child1_cuts = parent1.cuts[:crossover_point+1] + parent2.cuts[crossover_point+1:]
    child2_cuts = parent2.cuts[:crossover_point+1] + parent1.cuts[crossover_point+1:]
    
    child1_cuts = _smart_repair(child1_cuts, W)
    child2_cuts = _smart_repair(child2_cuts, W)
    
    return Individual(W, child1_cuts), Individual(W, child2_cuts)

def _uniform_crossover(parent1: Individual, parent2: Individual, W: int) -> Tuple[Individual, Individual]:
    child1_cuts = [0] + [0] * 6 + [W]
    child2_cuts = [0] + [0] * 6 + [W]
    
    for i in range(1, 7):
        if random.random() < 0.5:
            child1_cuts[i] = parent1.cuts[i]
            child2_cuts[i] = parent2.cuts[i]
        else:
            child1_cuts[i] = parent2.cuts[i]
            child2_cuts[i] = parent1.cuts[i]
    
    child1_cuts = _smart_repair(child1_cuts, W)
    child2_cuts = _smart_repair(child2_cuts, W)
    
    return Individual(W, child1_cuts), Individual(W, child2_cuts)

def _arithmetic_crossover(parent1: Individual, parent2: Individual, W: int) -> Tuple[Individual, Individual]:
    alpha = random.random()
    
    child1_cuts = [0] + [0] * 6 + [W]
    child2_cuts = [0] + [0] * 6 + [W]
    
    for i in range(1, 7):
        child1_cuts[i] = int(alpha * parent1.cuts[i] + (1 - alpha) * parent2.cuts[i])
        child2_cuts[i] = int((1 - alpha) * parent1.cuts[i] + alpha * parent2.cuts[i])
    
    child1_cuts = _smart_repair(child1_cuts, W)
    child2_cuts = _smart_repair(child2_cuts, W)
    
    return Individual(W, child1_cuts), Individual(W, child2_cuts)

def _crossover(parent1: Individual, parent2: Individual, W: int) -> Tuple[Individual, Individual]:
    strategy = random.random()
    
    if strategy < 0.4:
        return _single_point_crossover(parent1, parent2, W)
    elif strategy < 0.7:
        return _uniform_crossover(parent1, parent2, W)
    else:
        return _arithmetic_crossover(parent1, parent2, W)

def _gaussian_mutation(individual: Individual, extractor: EnhancedFeatureExtractor):
    mutated_cuts = individual.cuts.copy()

    mutation_points = random.sample(range(1, 7), random.randint(1, 3))
    
    for idx in mutation_points:
        sigma = individual.W * 0.05 
        noise = np.random.normal(0, sigma)
        mutated_cuts[idx] = int(mutated_cuts[idx] + noise)
    
    mutated_cuts = _smart_repair(mutated_cuts, individual.W)
    return Individual(individual.W, mutated_cuts)

def _projection_guided_mutation(individual: Individual, extractor: EnhancedFeatureExtractor):
    mutated_cuts = individual.cuts.copy()
    projection = extractor.projection

    mutation_points = random.sample(range(1, 7), random.randint(1, 2))
    
    for idx in mutation_points:
        current_pos = mutated_cuts[idx]

        search_range = min(20, individual.W // 8)
        start = max(mutated_cuts[idx-1] + 4, current_pos - search_range)
        end = min(mutated_cuts[idx+1] - 4, current_pos + search_range)
        
        if start < end and end < len(projection):
            window = projection[start:end+1]
            if len(window) > 0:
                min_idx = np.argmin(window)
                mutated_cuts[idx] = start + min_idx
    
    mutated_cuts = _smart_repair(mutated_cuts, individual.W)
    return Individual(individual.W, mutated_cuts)

def _adaptive_mutation(individual: Individual, extractor: EnhancedFeatureExtractor, generation: int):
    if individual.age > 10 or generation > 150:
        return _projection_guided_mutation(individual, extractor)
    else:
        return _gaussian_mutation(individual, extractor)

class Population:
    def __init__(self, pop_id: int, W: int, extractor: EnhancedFeatureExtractor):
        self.pop_id = pop_id
        self.W = W
        self.extractor = extractor
        self.individuals = []

        self._initialize_population()

        for individual in self.individuals:
            individual.evaluate(extractor)

        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)

        self.best_fitness = self.individuals[0].fitness
        self.avg_fitness = np.mean([ind.fitness for ind in self.individuals])
        self.generation = 0
    
    def _initialize_population(self):
        uniform_cuts = [0] + [int(i * self.W / 7) for i in range(1, 7)] + [self.W]
        self.individuals.append(Individual(self.W, uniform_cuts))

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
                
                projection_cuts = [0] + selected_valleys + [self.W]
                self.individuals.append(Individual(self.W, projection_cuts))

        while len(self.individuals) < POP_SIZE:
            self.individuals.append(Individual(self.W))
    
    def evolve(self):
        self.generation += 1

        elite_count = int(POP_SIZE * ELITE_RATE)
        elites = self.individuals[:elite_count]

        new_individuals = elites.copy()
        
        while len(new_individuals) < POP_SIZE:
            parent1 = _tournament_selection(self.individuals, TOURNAMENT_SIZE)
            parent2 = _tournament_selection(self.individuals, TOURNAMENT_SIZE)

            if random.random() < CROSSOVER_RATE:
                child1, child2 = _crossover(parent1, parent2, self.W)
            else:
                child1, child2 = parent1, parent2

            if random.random() < MUTATION_RATE:
                child1 = _adaptive_mutation(child1, self.extractor, self.generation)
            if random.random() < MUTATION_RATE:
                child2 = _adaptive_mutation(child2, self.extractor, self.generation)

            child1.evaluate(self.extractor)
            child2.evaluate(self.extractor)
            
            new_individuals.extend([child1, child2])

        new_individuals = new_individuals[:POP_SIZE]

        for individual in new_individuals:
            individual.age += 1

        if self.generation % LOCAL_SEARCH_FREQ == 0:
            for elite in elites[:3]:
                elite.local_search(self.extractor)

        self.individuals = new_individuals
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)

        self.best_fitness = self.individuals[0].fitness
        self.avg_fitness = np.mean([ind.fitness for ind in self.individuals])

def _population_migration(populations: List[Population]):
    migration_count = int(POP_SIZE * MIGRATION_RATE)
    
    for i in range(len(populations)):
        source_pop = populations[i]
        target_pop = populations[(i + 1) % len(populations)]

        migrants = source_pop.individuals[:migration_count]

        target_pop.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        target_pop.individuals[-migration_count:] = [
            Individual(target_pop.W, migrant.cuts) for migrant in migrants
        ]

        for individual in target_pop.individuals[-migration_count:]:
            individual.evaluate(target_pop.extractor)
        
        target_pop.individuals.sort(key=lambda ind: ind.fitness, reverse=True)

def _progress_bar(tag: str, current: int, total: int, extra_info: str = ""):

    bar_length = 35
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    percent = int(100 * current / total)
    
    sys.stdout.write(f"\r{tag} [{bar}] {percent:3d}% {extra_info}")
    sys.stdout.flush()

def search_best(*, img_path: str, iters: int = GENERATIONS) -> dict:
    img = cv2.imread(img_path)
    box = base.detect_plate_basic(img)
    if not box:
        print("[GA++] Cannot find license plate")
        return {}
    
    x, y, w, h = box
    W = w
    plate = base.ensure_plate_binary(img[y:y+h, x:x+w])

    extractor = EnhancedFeatureExtractor(plate)

    populations = [Population(i, W, extractor) for i in range(POPULATIONS)]
    
    global_best_cuts = None
    global_best_score = -1.0
    start_time = time.time()
    
    print(f"[GA++] Start superior，{POPULATIONS} populations，every population has {POP_SIZE} individual")

    for generation in range(min(iters, GENERATIONS)):
        if time.time() - start_time > TIME_LIMIT:
            break

        for population in populations:
            population.evolve()

        if generation % MIGRATION_FREQ == 0 and generation > 0:
            _population_migration(populations)

        for population in populations:
            if population.best_fitness > global_best_score:
                global_best_cuts = population.individuals[0].cuts.copy()
                global_best_score = population.best_fitness

        if generation % 15 == 0 or generation == 0:
            avg_pop_fitness = np.mean([pop.best_fitness for pop in populations])
            _progress_bar("[GA++]", generation + 1, min(iters, GENERATIONS), 
                        f"best={global_best_score:.4f} avg={avg_pop_fitness:.4f}")

    if global_best_cuts:
        print("\n[GA++] Final search...")

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
    print(f"[GA++] Superior end，use {execution_time:.2f}s，Best score: {global_best_score:.4f}")
    print(f"[GA++] Best cutting position: {global_best_cuts}")
    
    return {"cuts": global_best_cuts} if global_best_cuts else {}
