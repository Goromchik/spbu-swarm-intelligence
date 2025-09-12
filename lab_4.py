import tkinter as tk
from tkinter import messagebox
import numpy as np
import random


# Тестовая функция
def test_function(x1, x2):
    return 4 * ((x1 - 5) ** 2) + ((x2 - 6) ** 2)

# Генерация начальной популяции хромосом для вещественной кодировки
def generate_real_population(size, min_val, max_val):
    return np.random.uniform(min_val, max_val, (size, 2))


# Генерация начальной популяции хромосом для бинарной кодировки
def generate_binary_population(size, gene_length):
    return np.random.randint(2, size=(size, gene_length * 2))


# Декодирование бинарной хромосомы в вещественные значения x1 и x2
def decode_binary_chromosome(chromosome, min_val, max_val, gene_length):
    midpoint = len(chromosome) // 2
    x1_binary = chromosome[:midpoint]
    x2_binary = chromosome[midpoint:]

    x1 = int("".join(map(str, x1_binary)), 2)
    x2 = int("".join(map(str, x2_binary)), 2)

    x1 = min_val + (x1 / (2 ** gene_length - 1)) * (max_val - min_val)
    x2 = min_val + (x2 / (2 ** gene_length - 1)) * (max_val - min_val)
    return x1, x2


# Вычисление пригодности каждой хромосомы
def calculate_fitness(population, encoding, min_val, max_val, gene_length):
    if encoding == "real":
        return np.array([test_function(x1, x2) for x1, x2 in population])
    else:
        return np.array(
            [test_function(*decode_binary_chromosome(chromosome, min_val, max_val, gene_length)) for chromosome in
             population])


# Селекция
def selection(population, fitness, num_parents):
    parents = population[np.argsort(fitness)][:num_parents]
    return parents


# Одноточечный кроссинговер
def single_point_crossover(parents, offspring_size):
    offspring = np.empty(offspring_size, dtype=parents.dtype)
    crossover_point = np.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


# Однородный кроссинговер
def uniform_crossover(parents, offspring_size):
    offspring = np.empty(offspring_size, dtype=parents.dtype)
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        for j in range(offspring_size[1]):
            offspring[k, j] = parents[parent1_idx, j] if random.uniform(0, 1) < 0.5 else parents[parent2_idx, j]
    return offspring


# Мутация
def mutation(offspring_crossover, mutation_rate, encoding):
    for idx in range(offspring_crossover.shape[0]):
        if random.uniform(0, 1) < mutation_rate:
            if encoding == "real":
                offspring_crossover[idx, :] += np.random.uniform(-1.0, 1.0, 2)
            else:
                mutation_idx = random.randint(0, offspring_crossover.shape[1] - 1)
                offspring_crossover[idx, mutation_idx] ^= 1
    return offspring_crossover


# Генетический алгоритм
def genetic_algorithm(population_size, min_val, max_val, num_generations, mutation_rate, crossover_type, encoding,
                      gene_length=10):
    if encoding == "real":
        population = generate_real_population(population_size, min_val, max_val)
    else:
        population = generate_binary_population(population_size, gene_length)

    for generation in range(num_generations):
        fitness = calculate_fitness(population, encoding, min_val, max_val, gene_length)
        parents = selection(population, fitness, int(population_size / 2))

        # Выбор типа кроссинговера
        if crossover_type == "single_point":
            offspring_crossover = single_point_crossover(parents,
                                                         (population_size - parents.shape[0], parents.shape[1]))
        elif crossover_type == "uniform":
            offspring_crossover = uniform_crossover(parents, (population_size - parents.shape[0], parents.shape[1]))

        offspring_mutation = mutation(offspring_crossover, mutation_rate, encoding)
        population[:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation

    best_solution = population[np.argmin(calculate_fitness(population, encoding, min_val, max_val, gene_length))]
    if encoding == "binary":
        best_solution = decode_binary_chromosome(best_solution, min_val, max_val, gene_length)
    return best_solution, test_function(best_solution[0], best_solution[1])


# Функция для расчета хромосом
def calculate_chromosomes():
    try:
        population_size = int(population_size_entry.get())
        min_val = float(min_val_entry.get())
        max_val = float(max_val_entry.get())
        num_generations = int(num_generations_entry.get())
        mutation_rate = float(mutation_rate_entry.get()) / 100
        encoding = encoding_var.get()
        crossover_type = crossover_var.get()
        gene_length = 10  # Длина гена для бинарной кодировки

        best_solution, best_result = genetic_algorithm(population_size, min_val, max_val, num_generations,
                                                       mutation_rate, crossover_type, encoding, gene_length)
        result_label.config(text=f"Лучшее решение:\nX[1] = {best_solution[0]:.12f}\nX[2] = {best_solution[1]:.12f}")
        function_value_label.config(text=f"Значение функции: {best_result:.12f}")
        display_chromosomes()
    except ValueError:
        messagebox.showerror("Ошибка", "Введите корректные числовые значения.")


# Функция для отображения таблицы с хромосомами
def display_chromosomes():
    try:
        population_size = int(population_size_entry.get())
        min_val = float(min_val_entry.get())
        max_val = float(max_val_entry.get())
        encoding = encoding_var.get()
        population = generate_real_population(population_size, min_val,
                                              max_val) if encoding == "real" else generate_binary_population(
            population_size, 10)
        fitness = calculate_fitness(population, encoding, min_val, max_val, 10)
        chromosomes_text.delete(1.0, tk.END)
        for i, chromosome in enumerate(population):
            if encoding == "binary":
                x1, x2 = decode_binary_chromosome(chromosome, min_val, max_val, 10)
                chromosomes_text.insert(tk.END, f"{i + 1} {test_function(x1, x2):.2f} {x1:.2f} {x2:.2f}\n")
            else:
                x1, x2 = chromosome
                chromosomes_text.insert(tk.END, f"{i + 1} {test_function(x1, x2):.2f} {x1:.2f} {x2:.2f}\n")
    except ValueError:
        messagebox.showerror("Ошибка", "Введите корректные числовые значения.")


# Создание главного окна
root = tk.Tk()
root.title("Генетический алгоритм")

# Создание и размещение виджетов
tk.Label(root, text="Выберите кодировку:").grid(row=0, column=0, padx=10, pady=5)
encoding_var = tk.StringVar(value="real")
tk.Radiobutton(root, text="Вещественная", variable=encoding_var, value="real").grid(row=0, column=1, padx=5, pady=5)
tk.Radiobutton(root, text="Бинарная", variable=encoding_var, value="binary").grid(row=0, column=2, padx=5, pady=5)

tk.Label(root, text="Выберите кроссинговер:").grid(row=1, column=0, padx=10, pady=5)
crossover_var = tk.StringVar(value="single_point")
tk.Radiobutton(root, text="Одноточечный", variable=crossover_var, value="single_point").grid(row=1, column=1, padx=5,
                                                                                             pady=5)
tk.Radiobutton(root, text="Однородный", variable=crossover_var, value="uniform").grid(row=1, column=2, padx=5, pady=5)

tk.Label(root, text="Вероятность мутации, %:").grid(row=2, column=0, padx=10, pady=5)
mutation_rate_entry = tk.Entry(root)
mutation_rate_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Количество хромосом:").grid(row=3, column=0, padx=10, pady=5)
population_size_entry = tk.Entry(root)
population_size_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Минимальное значение гена:").grid(row=4, column=0, padx=10, pady=5)
min_val_entry = tk.Entry(root)
min_val_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Максимальное значение гена:").grid(row=5, column=0, padx=10, pady=5)
max_val_entry = tk.Entry(root)
max_val_entry.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Количество поколений:").grid(row=6, column=0, padx=10, pady=5)
num_generations_entry = tk.Entry(root)
num_generations_entry.grid(row=6, column=1, padx=10, pady=5)

calculate_chromosomes_button = tk.Button(root, text="Рассчитать хромосомы", command=calculate_chromosomes)
calculate_chromosomes_button.grid(row=7, column=0, columnspan=2, pady=10)

result_label = tk.Label(root, text="Лучшее решение:")
result_label.grid(row=8, column=0, columnspan=2, pady=10)

function_value_label = tk.Label(root, text="Значение функции:")
function_value_label.grid(row=9, column=0, columnspan=2, pady=10)

chromosomes_text = tk.Text(root, width=50, height=10)
chromosomes_text.grid(row=10, column=0, columnspan=3, pady=10)

# Запуск главного цикла
root.mainloop()
