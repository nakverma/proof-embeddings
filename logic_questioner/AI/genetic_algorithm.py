import json
import numpy as np
from random import random, randint
from test_astar_search import get_score_simple
from astar_heuristics import *
from time import time
from multiprocessing.pool import ThreadPool


class GeneticAlgorithm:

    def __init__(self, heuristics, ranges):
        self.heuristics = heuristics
        self.pop_size = 0
        self.ranges = ranges
        self.population = None
        self.is_elite = None
        self.scores = None, None
        self.generation_count = 0
        self.best_individual = None
        self.best_score = 0

    def create_population(self):
        pop = np.array([
            np.concatenate(
                [np.random.uniform(low=r[0], high=r[1], size=1) for r in self.ranges]
            ) for _ in range(self.pop_size)
        ])
        return pop, np.zeros(pop.shape[0])

    def crossover(self, i1, i2, pc):
        if random() > pc:
            return np.copy(i1), np.copy(i2)
        start, end = np.random.choice(len(i1)-1, size=2, replace=False)
        start, end = (end, start) if start > end else (start, end)
        c1 = np.concatenate((i1[:start], i2[start: end], i1[end:]))
        c2 = np.concatenate((i2[:start], i1[start: end], i2[end:]))
        return np.array([c1, c2])

    def point_mutation(self, individual, pm):
        if random() > pm:
            return np.copy(individual)
        rand_idx = randint(0, len(individual)-1)
        individual[rand_idx] = np.random.uniform(low=self.ranges[rand_idx][0], high=self.ranges[rand_idx][1], size=1)[0]
        return individual

    def get_score(self, idx, questions, max_timeout):
        print(".", end="")
        if self.is_elite[idx]:
            return self.scores[idx]
        gh = GeneHeuristic(self.heuristics, self.population[idx])
        return get_score_simple(gh.gene_meta_dist, questions, max_timeout)

    def update_scores(self, questions, max_timeout):
        self.scores = np.array([self.get_score(i, questions, max_timeout) for i in range(len(self.population))])
        best_idx = np.argmax(self.scores)
        if self.scores[best_idx] > self.best_score:
            self.best_individual, self.best_score = self.population[best_idx], self.scores[best_idx]

    def tournament_selection(self, k=3):
        idxs = np.random.choice(self.pop_size, k, replace=False)
        pick = max(idxs, key=lambda i: self.scores[i])
        return pick

    def run_generation(self, questions, elitism, pc, pm, max_timeout):
        parent_idxs = [self.tournament_selection() for _ in range(self.pop_size)]
        parents, parent_scores = self.population[parent_idxs], self.scores[parent_idxs]
        children = []

        for i in range(len(parents)-1):
            for c in self.crossover(parents[i], parents[i+1], pc):
                mc = self.point_mutation(c, pm)
                children.append(mc)

        if elitism > 0:
            sorted_order = np.argsort(self.scores)[::-1]
            sorted_pop, sorted_scores = self.population[sorted_order], self.scores[sorted_order]
            elite = sorted_pop[:elitism]
        else:
            elite = []
            sorted_scores = []

        self.population = np.concatenate((np.copy(elite[:elitism]), children[:self.pop_size-elitism]))
        self.scores = np.concatenate((sorted_scores[:elitism], self.scores[elitism:]))  # first few not reevaluated
        self.update_scores(questions, max_timeout)

    def train(self, population_size, questions, num_generations=10, elitism=1, pc=0.8, pm=0.2, max_timeout=1):
        begin_time = time()

        if self.population is None:
            start = time()
            self.pop_size = population_size
            self.population, self.scores = self.create_population()
            self.is_elite = [False] * self.pop_size
            self.update_scores(questions, max_timeout)
            end = time()
            print(f"\nBaseline | {end-start:.4f} seconds | "
                  f"\tAvg Score: {np.mean(self.scores)}\tBest Score: {self.best_score}")
            print(f"Scores: {self.scores}")
            print(f"Best Weights: {self.best_individual}\n")

        self.is_elite = [True]*elitism + [False]*(self.pop_size-elitism)
        for g in range(num_generations):
            start = time()
            self.run_generation(questions, elitism, pc, pm, max_timeout)
            end = time()
            print(f"\nGeneration {g+1}/{num_generations} | {end-start:.4f} seconds | "
                  f"\tAvg Score: {np.mean(self.scores)}\tBest Score: {self.best_score}")
            print(f"Scores: {self.scores}")
            print(f"Best Weights: {self.best_individual}\n")

        end_time = time()
        print(f"Total running time: {end_time-begin_time:.4f} seconds.")
        return GeneHeuristic(self.heuristics, self.best_individual)

    def save(self):
        pass

    def load(self):
        pass


def base_ga_train():
    heuristics = [levenshtein_distance, unitary_distance, variable_mismatch] + RuleDists().all_dists
    ranges = [(0, 10)] + [(-10, 10)] * (len(heuristics) - 1)
    with open("../questions.json", "r") as qf:
        questions = json.load(qf)['questions']

    ga = GeneticAlgorithm(heuristics, ranges)
    gh = ga.train(population_size=10, questions=questions[:5], num_generations=10, max_timeout=1, elitism=1)
    return gh


if __name__ == "__main__":
    gh = base_ga_train()
    print(gh.weights)
    gh.save("gene_weights_var.txt")




