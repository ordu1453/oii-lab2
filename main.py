import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# --- Определение переменных ---
error = np.arange(-10, 10.1, 0.1)
delta = np.arange(-2, 2.1, 0.1)
speed = np.arange(0, 20.1, 0.1)

# --- Функции принадлежности ---
error_too_close = fuzz.trapmf(error, [-10, -10, -6, -2])
error_normal = fuzz.trimf(error, [-3, 0, 3])
error_far = fuzz.trapmf(error, [2, 6, 10, 10])

delta_approaching = fuzz.trapmf(delta, [-2, -2, -0.5, 0])
delta_steady = fuzz.trimf(delta, [-0.3, 0, 0.3])
delta_moving_away = fuzz.trapmf(delta, [0, 0.5, 2, 2])

speed_slow = fuzz.trapmf(speed, [0, 0, 2, 6])
speed_medium = fuzz.trimf(speed, [4, 8, 12])
speed_fast = fuzz.trapmf(speed, [10, 14, 20, 20])

# --- Правила ---
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

# --- Универсумы и словари функций ---
error_mfs = {'too_close': error_too_close, 'normal': error_normal, 'far': error_far}
delta_mfs = {'approaching': delta_approaching, 'steady': delta_steady, 'moving_away': delta_moving_away}
speed_mfs = {'slow': speed_slow, 'medium': speed_medium, 'fast': speed_fast}

# --- Симуляция ---
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
    v_leader[i] = 6
    if 30 < time[i] < 60:
        v_leader[i] = 11
    elif time[i] > 60:
        v_leader[i] = 6
    x_leader[i] = x_leader[i-1] + v_leader[i] * dt

    # Ошибки
    distance = x_leader[i] - x_auto[i-1]
    e = distance - desired_distance
    de = (e - error_list[i-1]) / dt if i > 1 else 0

    # --- Фаззификация ---
    mu_error = {label: fuzz.interp_membership(error, mf, np.clip(e, -10, 10)) for label, mf in error_mfs.items()}
    mu_delta = {label: fuzz.interp_membership(delta, mf, np.clip(de, -2, 2)) for label, mf in delta_mfs.items()}

    # --- Импликация Ларсена с α = product ---
    aggregated = np.zeros_like(speed)
    for err_label, d_label, s_label in rules:
        alpha = mu_error[err_label] * mu_delta[d_label]  # <-- изменено: произведение вместо min()
        implied = alpha * speed_mfs[s_label]             # Larsen implication
        aggregated = np.fmax(aggregated, implied)        # объединение правил (max)

    # --- Дефаззификация ---
    try:
        v_auto[i] = fuzz.defuzz(speed, aggregated, 'centroid')
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
