import numpy as np
import random
from math import sqrt
import matplotlib.pyplot as plt


class GeneticAlgorithmTSP:
    def __init__(self, cities, population_size=100, elite_size=20, mutation_rate=0.01, generations=500):
        self.cities = cities
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations

        self.distance_matrix = self._create_distance_matrix()

    def _create_distance_matrix(self):
        n = len(self.cities)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                distance = sqrt((self.cities[i][0] - self.cities[j][0]) ** 2 +
                                (self.cities[i][1] - self.cities[j][1]) ** 2)
                matrix[i][j] = distance
                matrix[j][i] = distance
        return matrix

    def _calculate_route_distance(self, route):
        """Вычисляет длину маршрута (используется во всех расчетах)"""
        distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            distance += self.distance_matrix[from_city][to_city]
        return distance

    def _initial_population(self):
        """Создает начальную популяцию случайных маршрутов"""
        population = []
        city_list = list(range(len(self.cities)))

        for _ in range(self.population_size):
            individual = city_list.copy()
            random.shuffle(individual)
            population.append(individual)
        return population

    def _calculate_fitness(self, individual):
        """Вычисляет приспособленность особи (обратную длине маршрута)"""
        return 1 / self._calculate_route_distance(individual)

    def _rank_population(self, population):
        """Ранжирует популяцию по приспособленности"""
        fitness_results = {}
        for i, individual in enumerate(population):
            fitness_results[i] = self._calculate_fitness(individual)
        return sorted(fitness_results.items(), key=lambda x: x[1], reverse=True)

    def _selection(self, pop_ranked):
        """Отбор особей для создания следующего поколения"""
        selection_results = []
        total_fitness = sum([item[1] for item in pop_ranked])
        probs = [item[1] / total_fitness for item in pop_ranked]

        # Элитные особи
        for i in range(self.elite_size):
            selection_results.append(pop_ranked[i][0])

        # Рулеточный отбор для остальных
        for _ in range(self.elite_size, self.population_size):
            pick = random.random()
            current = 0
            for i, prob in enumerate(probs):
                current += prob
                if current > pick:
                    selection_results.append(pop_ranked[i][0])
                    break
        return selection_results

    def _crossover(self, parent1, parent2):
        """Одноточечный кроссовер с исправлением дублирования"""
        child = [None] * len(parent1)
        crossover_point = random.randint(0, len(parent1) - 1)
        child[:crossover_point] = parent1[:crossover_point]
        remaining_genes = [gene for gene in parent2 if gene not in child]
        child[crossover_point:] = remaining_genes[:len(parent1) - crossover_point]
        return child

    def _mutate(self, individual):
        """Мутация - перестановка двух случайных городов"""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(individual) - 1)
                individual[i], individual[j] = individual[j], individual[i]
        return individual

    def _create_new_generation(self, current_pop, selection_results):
        new_population = []

        # Элитные особи
        for i in range(self.elite_size):
            new_population.append(current_pop[selection_results[i]])

        # Скрещивание
        for i in range(self.elite_size, self.population_size, 2):
            parent1 = current_pop[selection_results[i]]
            parent2 = current_pop[selection_results[i + 1]] if i + 1 < len(selection_results) else current_pop[selection_results[0]]

            child1 = self._crossover(parent1, parent2)
            child2 = self._crossover(parent2, parent1)

            new_population.extend([child1, child2])

        # Мутация
        for i in range(len(new_population)):
            new_population[i] = self._mutate(new_population[i])

        return new_population

    def run(self):
        population = self._initial_population()
        best_individual = None
        best_distance = float('inf')
        progress = []

        for generation in range(self.generations):
            ranked_population = self._rank_population(population)
            selection_results = self._selection(ranked_population)
            population = self._create_new_generation(population, selection_results)

            current_best_index = ranked_population[0][0]
            current_best_individual = population[current_best_index]
            current_best_distance = self._calculate_route_distance(current_best_individual)

            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_individual = current_best_individual.copy()

            progress.append(best_distance)

            if generation % 50 == 0:
                print(f"Generation {generation}: Best Distance = {best_distance:.2f}")

        return best_individual, best_distance, progress

    def plot_progress(self, progress):
        plt.figure(figsize=(10, 5))
        plt.plot(progress)
        plt.title('Optimization Progress')
        plt.xlabel('Generation')
        plt.ylabel('Best Route Distance')
        plt.grid(True)
        plt.show()

    def plot_route(self, route):
        route_distance = self._calculate_route_distance(route)
        x = [self.cities[i][0] for i in route] + [self.cities[route[0]][0]]
        y = [self.cities[i][1] for i in route] + [self.cities[route[0]][1]]

        plt.figure(figsize=(10, 8))
        plt.plot(x, y, 'o-', linewidth=2, markersize=8)

        for i, city in enumerate(self.cities):
            plt.text(city[0], city[1], str(i), fontsize=12, ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        plt.title(f"TSP Route - Distance: {route_distance:.2f}", fontsize=14)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    random.seed(42)

    num_cities = 15
    cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_cities)]

    ga = GeneticAlgorithmTSP(cities, population_size=100, elite_size=20,
                             mutation_rate=0.02, generations=300)

    best_route, best_distance, progress = ga.run()

    print("\nFinal Results:")
    print(f"Best Route: {best_route}")
    print(f"Best Distance: {best_distance:.2f}")

    calculated_distance = ga._calculate_route_distance(best_route)
    print(f"Calculated Distance: {calculated_distance:.2f}")
    assert abs(best_distance - calculated_distance) < 1e-6, "Distance calculation mismatch!"

    ga.plot_progress(progress)
    ga.plot_route(best_route)