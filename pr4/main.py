import numpy as np
import matplotlib.pyplot as plt


dim = 2                  # Размерность
bounds = np.array([[-2, 2], [-2, 2]])  # Границы поиска
n_saplings = 50          # Число саженцев
max_iter = 200           # Число итераций
epsilon = 0.1            # Порог близости для прививки


def rosenbrock(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))


def initialize_population():
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.rand(n_saplings, dim)


def crossover(si, sj):
    r = np.sqrt(np.sum((bounds[:, 1] - bounds[:, 0])**2))
    distance = np.linalg.norm(si - sj) / r
    xi_cross = np.random.rand(dim) < (1 - distance)
    new_si = np.where(xi_cross, sj, si)
    new_sj = np.where(xi_cross, si, sj)
    return new_si, new_sj


def branching(si, k):
    l = np.random.randint(0, dim)
    if l == k:
        return si
    prob = 1 - 1 / (1 + (l - k)**2)
    if np.random.rand() < prob:
        si[l] = np.random.uniform(bounds[l, 0], bounds[l, 1])
    return si


def vaccinate(si, sj):
    mu = np.sum(np.abs(si - sj) / (bounds[:, 1] - bounds[:, 0]))
    if mu > epsilon * dim:
        mask = np.abs(si - sj) / (bounds[:, 1] - bounds[:, 0]) > epsilon * dim
        new_si = np.where(mask, sj, si)
        new_sj = np.where(mask, si, sj)
        return new_si, new_sj
    return si, sj


def ssg_rosenbrock():
    population = initialize_population()
    best_solution = None
    best_fitness = float('inf')
    history = []

    for iteration in range(max_iter):
        for i in range(n_saplings):
            j = np.random.randint(0, n_saplings)
            if i != j:
                population[i], population[j] = crossover(population[i], population[j])
            k = np.random.randint(0, dim)
            population[i] = branching(population[i], k)
            j = np.random.randint(0, n_saplings)
            if i != j:
                population[i], population[j] = vaccinate(population[i], population[j])

        fitness = np.array([rosenbrock(x) for x in population])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_solution = population[best_idx]
        history.append(best_fitness)

    return best_solution, best_fitness, history


best_x, best_f, history = ssg_rosenbrock()
print(f"Лучшее решение: x = {best_x}, f(x) = {best_f}")

plt.plot(history)
plt.xlabel("Итерация")
plt.ylabel("Лучшее значение f(x)")
plt.title("Оптимизация функции Розенброка с SSG")
plt.grid()
plt.show()