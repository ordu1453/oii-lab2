import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# -------------------------------
# 1. Нечеткие переменные
# -------------------------------
distance_to_leader = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')
v_autopilot = ctrl.Consequent(np.arange(0, 101, 1), 'v_autopilot')
v_leader = ctrl.Antecedent(np.arange(0, 101, 1), 'v_leader')

# Функции принадлежности дистанции
distance_to_leader['TooClose'] = fuzz.trapmf(distance_to_leader.universe, [0,0,10,30])
distance_to_leader['Safe']     = fuzz.trimf(distance_to_leader.universe, [20,50,80])
distance_to_leader['Far']      = fuzz.trapmf(distance_to_leader.universe, [70,90,100,100])

# Функции принадлежности скорости автопилота
v_autopilot['Slow']   = fuzz.trapmf(v_autopilot.universe, [0,0,20,40])
v_autopilot['Medium'] = fuzz.trimf(v_autopilot.universe, [30,50,70])
v_autopilot['Fast']   = fuzz.trapmf(v_autopilot.universe, [60,80,100,100])

# Функции принадлежности скорости лидера
v_leader['Zero']   = fuzz.trapmf(v_leader.universe, [0,0,0,5])
v_leader['Medium'] = fuzz.trimf(v_leader.universe, [0,50,80])
v_leader['Fast']   = fuzz.trapmf(v_leader.universe, [60,80,100,100])

# Правила Ларсена (для автопилота по дистанции)
rules = [
    ('TooClose','Slow'),
    ('Safe','Medium'),
    ('Far','Fast')
]

def larsen_rule(firing_strength, consequent_mf):
    return firing_strength * consequent_mf

# -------------------------------
# 2. Функция расчета скорости автопилота
# -------------------------------
def compute_autopilot_speed(distance_input, leader_speed):
    # Фаззификация дистанции
    dist_level = {key: fuzz.interp_membership(distance_to_leader.universe,
                                              distance_to_leader[key].mf,
                                              distance_input)
                  for key in distance_to_leader.terms.keys()}
    # Фаззификация скорости лидера
    v_level = {key: fuzz.interp_membership(v_leader.universe,
                                           v_leader[key].mf,
                                           leader_speed)
               for key in v_leader.terms.keys()}
    
    # Агрегация (product Ларсена)
    output_aggregated = np.zeros_like(v_autopilot.universe)
    for dist_term, auto_term in rules:
        for v_term in v_level:
            alpha = dist_level[dist_term] * v_level[v_term]
            output_aggregated = np.fmax(output_aggregated, larsen_rule(alpha, v_autopilot[auto_term].mf))
    
    v_output = fuzz.defuzz(v_autopilot.universe, output_aggregated, 'centroid')
    return v_output, dist_level, output_aggregated

# -------------------------------
# 3. GUI
# -------------------------------
root = tk.Tk()
root.title("Dynamic Fuzzy Autopilot Simulation")

# Слайдер скорости лидера
slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL,
                  length=400, label='Leader speed')
slider.set(50)
slider.pack()

# Метки для отображения дистанции и степеней принадлежности
distance_label = tk.Label(root, text="Distance: 0.0")
distance_label.pack()
membership_label = tk.Label(root, text="Memberships: TooClose=0.0, Safe=0.0, Far=0.0")
membership_label.pack()

# График
fig, ax = plt.subplots(figsize=(8,4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Начальные позиции
leader_pos = 0.0
auto_pos = -20.0
time_step = 0.1

leader_positions = []
autopilot_positions = []

# -------------------------------
# 4. Анимация движения
# -------------------------------
def update(frame):
    global leader_pos, auto_pos
    v_input = slider.get()
    
    # Лидер движется с заданной скоростью
    leader_pos += v_input * time_step
    
    # Расстояние до лидера
    distance = leader_pos - auto_pos
    distance_clamped = max(0, min(100, distance))
    
    # Скорость автопилота
    v_auto, dist_level, output_agg = compute_autopilot_speed(distance_clamped, v_input)
    auto_pos += v_auto * time_step
    
    # Сохраняем позиции
    leader_positions.append(leader_pos)
    autopilot_positions.append(auto_pos)
    
    # Обновляем график
    ax.clear()
    ax.plot(leader_positions, label='Leader', color='blue')
    ax.plot(autopilot_positions, label='Autopilot', color='red')
    ax.set_ylim(0, max(max(leader_positions)+20, 100))
    ax.set_xlim(0, max(len(leader_positions),100))
    ax.set_ylabel("Position")
    ax.set_xlabel("Time step")
    ax.legend()
    ax.set_title(f"Leader speed: {v_input:.1f}, Autopilot speed: {v_auto:.1f}")
    
    # Обновляем метки
    distance_label.config(text=f"Distance: {distance:.2f}")
    membership_label.config(text=f"Memberships: TooClose={dist_level['TooClose']:.2f}, "
                                 f"Safe={dist_level['Safe']:.2f}, Far={dist_level['Far']:.2f}")
    
    canvas.draw()

ani = FuncAnimation(fig, update, interval=100)

root.mainloop()
