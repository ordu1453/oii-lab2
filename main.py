import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- Определение переменных ---
error = ctrl.Antecedent(np.arange(-10, 10.1, 0.1), 'error')
delta = ctrl.Antecedent(np.arange(-2, 2.1, 0.1), 'delta')
speed = ctrl.Consequent(np.arange(0, 20.1, 0.1), 'speed')

# --- Функции принадлежности ---
error['too_close'] = fuzz.trapmf(error.universe, [-10, -10, -6, -2])
error['normal'] = fuzz.trimf(error.universe, [-3, 0, 3])
error['far'] = fuzz.trapmf(error.universe, [2, 6, 10, 10])

delta['approaching'] = fuzz.trapmf(delta.universe, [-2, -2, -0.5, 0])
delta['steady'] = fuzz.trimf(delta.universe, [-0.3, 0, 0.3])
delta['moving_away'] = fuzz.trapmf(delta.universe, [0, 0.5, 2, 2])

speed['slow'] = fuzz.trapmf(speed.universe, [0, 0, 2, 6])
speed['medium'] = fuzz.trimf(speed.universe, [4, 8, 12])
speed['fast'] = fuzz.trapmf(speed.universe, [10, 14, 20, 20])

# --- Определение правил ---
rules = [
    ('too_close', 'approaching', 'slow'),
    ('too_close', 'steady', 'slow'),
    ('too_close', 'moving_away', 'medium'),
    ('normal', 'approaching', 'slow'),
    ('normal', 'steady', 'medium'),
    ('normal', 'moving_away', 'fast'),
    ('far', 'approaching', 'medium'),
    ('far', 'steady', 'fast'),
    ('far', 'moving_away', 'fast'),
]

# --- Симуляция системы ---
dt = 0.1
T = 100
time = np.arange(0, T, dt)

v_leader = np.zeros_like(time)
v_auto = np.zeros_like(time)
x_leader = np.zeros_like(time)
x_auto = np.zeros_like(time)
error_list = np.zeros_like(time)

x_leader[0] = 0
x_auto[0] = -10
v_auto[0] = 5
v_leader[0] = 6
desired_distance = 10

for i in range(1, len(time)):
    # Скорость лидера
    v_leader[i] = 5
    if 30 < time[i] < 60:
        v_leader[i] = 7
    elif time[i] > 60:
        v_leader[i] = 11
    x_leader[i] = x_leader[i-1] + v_leader[i] * dt

    # Ошибки
    distance = x_leader[i] - x_auto[i-1]
    e = distance - desired_distance
    de = (e - error_list[i-1]) / dt if i > 1 else 0

    # --- Fuzzy inference по Ларсену ---
    mu_error = {label: fuzz.interp_membership(error.universe, error[label].mf, np.clip(e, -10, 10)) 
                for label in error.terms}
    mu_delta = {label: fuzz.interp_membership(delta.universe, delta[label].mf, np.clip(de, -2, 2)) 
                for label in delta.terms}

    aggregated = np.zeros_like(speed.universe)

    for err_label, d_label, s_label in rules:
        alpha = min(mu_error[err_label], mu_delta[d_label])
        # Larsen implication: multiply membership function by firing strength
        implied = alpha * speed[s_label].mf
        aggregated = np.fmax(aggregated, implied)

    # Дефаззификация (центроид)
    try:
        v_auto[i] = fuzz.defuzz(speed.universe, aggregated, 'centroid')
    except:
        v_auto[i] = v_auto[i-1]

    x_auto[i] = x_auto[i-1] + v_auto[i] * dt
    error_list[i] = abs(e)

# --- Графики ---
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, x_leader, label='Лидер', linewidth=2)
plt.plot(time, x_auto, label='Автопилот', linewidth=2)
plt.title('Координаты автомобилей')
plt.xlabel('Время, с')
plt.ylabel('X, м')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, error_list, color='red', label='Ошибка дистанции')
plt.axhline(0, color='black', linestyle='--')
plt.title('Ошибка поддержания дистанции')
plt.xlabel('Время, с')
plt.ylabel('Ошибка (м)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

mse = np.mean(error_list**2)
print(f"Среднеквадратичная ошибка дистанции: {mse:.3f}")
