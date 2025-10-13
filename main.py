import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# -------------------------------
# 1. Определяем нечёткие переменные
# -------------------------------
distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')
v_autopilot = ctrl.Consequent(np.arange(0, 101, 1), 'v_autopilot')

# Функции принадлежности для расстояния
distance['TooClose'] = fuzz.trapmf(distance.universe, [0, 0, 10, 30])
distance['Safe']     = fuzz.trimf(distance.universe, [20, 50, 80])
distance['Far']      = fuzz.trapmf(distance.universe, [70, 90, 100, 100])

# Функции принадлежности для скорости
v_autopilot['Slow']   = fuzz.trapmf(v_autopilot.universe, [0, 0, 20, 40])
v_autopilot['Medium'] = fuzz.trimf(v_autopilot.universe, [30, 50, 70])
v_autopilot['Fast']   = fuzz.trapmf(v_autopilot.universe, [60, 80, 100, 100])

# Правила Ларсена
rules = [
    ('TooClose', 'Slow'),
    ('Safe', 'Medium'),
    ('Far', 'Fast')
]

def larsen_rule(firing_strength, consequent_mf):
    return firing_strength * consequent_mf

# -------------------------------
# 2. Функция расчета скорости автопилота
# -------------------------------
def compute_autopilot_speed(dist_input):
    distance_level = {key: fuzz.interp_membership(distance.universe, distance[key].mf, dist_input)
                      for key in distance.terms.keys()}
    
    output_aggregated = np.zeros_like(v_autopilot.universe)
    for dist_term, v_term in rules:
        alpha = distance_level[dist_term]
        output_aggregated = np.fmax(output_aggregated, larsen_rule(alpha, v_autopilot[v_term].mf))
    
    v_output = fuzz.defuzz(v_autopilot.universe, output_aggregated, 'centroid')
    
    return v_output, output_aggregated

# -------------------------------
# 3. GUI с Tkinter
# -------------------------------
def update_graph(val):
    dist_input = slider.get()
    v_output, output_aggregated = compute_autopilot_speed(dist_input)
    
    ax.clear()
    ax.plot(v_autopilot.universe, output_aggregated, 'r', linewidth=2)
    ax.set_title(f"Distance: {dist_input:.1f}, Autopilot speed: {v_output:.2f}")
    ax.set_xlabel("Autopilot speed")
    ax.set_ylabel("Membership")
    canvas.draw()

root = tk.Tk()
root.title("Autopilot Fuzzy Control (Larsen)")

slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL,
                  length=400, label='Distance to leader', command=update_graph)
slider.set(50)
slider.pack()

fig, ax = plt.subplots(figsize=(6,3))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Инициализация графика
update_graph(slider.get())

root.mainloop()
