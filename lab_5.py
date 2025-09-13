import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Тестовая функция
def test_function(x1, x2):
    return 4 * ((x1 - 5) ** 2) + ((x2 - 6) ** 2)


# Класс Particle (Частица)
class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_value = float('inf')


# Роевой алгоритм
def pso(num_particles, num_iterations, w, c1, c2, max_velocity, function, bounds):
    particles = [Particle(np.random.uniform(bounds[0], bounds[1], 2),
                          np.random.uniform(-1, 1, 2)) for _ in range(num_particles)]

    global_best_position = np.random.uniform(bounds[0], bounds[1], 2)
    global_best_value = float('inf')

    for _ in range(num_iterations):
        for particle in particles:
            # Вычисление приспособленности текущей позиции
            fitness = function(particle.position[0], particle.position[1])

            # Обновление личного лучшего результата
            if fitness < particle.best_value:
                particle.best_position = particle.position.copy()
                particle.best_value = fitness

            # Обновление глобального лучшего результата
            if fitness < global_best_value:
                global_best_position = particle.position.copy()
                global_best_value = fitness

        # Обновление скорости и позиции
        for particle in particles:
            inertia = w * particle.velocity
            cognitive = c1 * np.random.rand() * (particle.best_position - particle.position)
            social = c2 * np.random.rand() * (global_best_position - particle.position)
            particle.velocity = inertia + cognitive + social

            # Применение ограничения скорости
            if max_velocity is not None:
                for i in range(len(particle.velocity)):
                    if abs(particle.velocity[i]) > max_velocity:
                        particle.velocity[i] = max_velocity * np.sign(particle.velocity[i])

            # Обновление позицию
            particle.position += particle.velocity
            # Ограничение позиции в пределах заданного диапазона
            particle.position = np.clip(particle.position, bounds[0], bounds[1])

    return global_best_position, global_best_value, particles


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Particle Swarm Optimization")

        # Фрейм параметров
        parameters_frame = tk.Frame(root)
        parameters_frame.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(parameters_frame, text="Количество частиц:").grid(row=0, column=0, sticky="e")
        self.num_particles_entry = tk.Entry(parameters_frame)
        self.num_particles_entry.insert(0, "300")
        self.num_particles_entry.grid(row=0, column=1)

        tk.Label(parameters_frame, text="Количество итераций:").grid(row=1, column=0, sticky="e")
        self.num_iterations_entry = tk.Entry(parameters_frame)
        self.num_iterations_entry.insert(0, "100")
        self.num_iterations_entry.grid(row=1, column=1)

        tk.Label(parameters_frame, text="Коэффициент инерции (w):").grid(row=2, column=0, sticky="e")
        self.w_entry = tk.Entry(parameters_frame)
        self.w_entry.insert(0, "0.3")
        self.w_entry.grid(row=2, column=1)

        tk.Label(parameters_frame, text="Коэфф. собственного лучшего значения (c1):").grid(row=3, column=0, sticky="e")
        self.c1_entry = tk.Entry(parameters_frame)
        self.c1_entry.insert(0, "2.0")
        self.c1_entry.grid(row=3, column=1)

        tk.Label(parameters_frame, text="Коэфф. глобального лучшего значения (c2):").grid(row=4, column=0, sticky="e")
        self.c2_entry = tk.Entry(parameters_frame)
        self.c2_entry.insert(0, "5.0")
        self.c2_entry.grid(row=4, column=1)

        tk.Label(parameters_frame, text="Макс. скорость частиц:").grid(row=5, column=0, sticky="e")
        self.max_velocity_entry = tk.Entry(parameters_frame)
        self.max_velocity_entry.insert(0, "10.0")
        self.max_velocity_entry.grid(row=5, column=1)

        self.limit_speed_var = tk.IntVar()
        self.limit_speed_check = tk.Checkbutton(parameters_frame, text="Ограничение скорости", variable=self.limit_speed_var)
        self.limit_speed_check.grid(row=6, column=0, columnspan=2, pady=5)

        self.start_button = tk.Button(parameters_frame, text="Рассчитать", command=self.start_pso)
        self.start_button.grid(row=7, column=0, columnspan=2, pady=10)

        self.result_label = tk.Label(parameters_frame, text="Лучшее решение:", font=("Arial", 10, "bold"))
        self.result_label.grid(row=8, column=0, columnspan=2)
        self.function_value_label = tk.Label(parameters_frame, text="Значение функции:", font=("Arial", 10, "bold"))
        self.function_value_label.grid(row=9, column=0, columnspan=2)

        # Фрейм графика
        plot_frame = tk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack()

    def start_pso(self):
        try:
            num_particles = int(self.num_particles_entry.get())
            num_iterations = int(self.num_iterations_entry.get())
            w = float(self.w_entry.get())
            c1 = float(self.c1_entry.get())
            c2 = float(self.c2_entry.get())
            max_velocity = float(self.max_velocity_entry.get()) if self.limit_speed_var.get() else None

            # Запуск PSO
            best_position, best_value, particles = pso(
                num_particles, num_iterations, w, c1, c2, max_velocity, test_function, bounds=(-500, 500)
            )

            # Обновление отображения результатов
            self.result_label.config(
                text=f"Лучшее решение:\nX[0] = {best_position[0]:.6f}\nX[1] = {best_position[1]:.6f}")
            self.function_value_label.config(text=f"Значение функции: {best_value:.6f}")

            # Построение графика частиц
            self.ax.clear()
            self.ax.set_title("Решения")
            self.ax.set_xlim(-500, 500)
            self.ax.set_ylim(-500, 500)
            for particle in particles:
                self.ax.plot(particle.position[0], particle.position[1], 'ko')
            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числовые значения.")


# Основное приложение
root = tk.Tk()
app = GUI(root)
root.mainloop()