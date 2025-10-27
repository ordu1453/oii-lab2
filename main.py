import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# === 1. Нечёткие переменные ===
error = ctrl.Antecedent(np.arange(-20, 20.1, 0.5), 'error')   # м
delta = ctrl.Antecedent(np.arange(-10, 10.1, 0.5), 'delta')   # м/с
accel = ctrl.Consequent(np.arange(-3, 3.1, 0.1), 'accel', defuzzify_method='centroid') 

# === 2. Функции принадлежности ===
error['too_close'] = fuzz.trapmf(error.universe, [-20, -20, -10, -3])
error['normal']    = fuzz.trimf(error.universe, [-4, 0, 4])  
error['far']       = fuzz.trapmf(error.universe, [3, 10, 20, 20])

delta['approaching']   = fuzz.trapmf(delta.universe, [-10, -10, -2, 0])
delta['steady']        = fuzz.trimf(delta.universe, [-1, 0, 1])
delta['moving_away']   = fuzz.trapmf(delta.universe, [0, 2, 10, 10])

accel['brake']       = fuzz.trapmf(accel.universe, [-3, -3, -1.5, -0.5])
accel['hold']        = fuzz.trimf(accel.universe, [-0.5, 0, 0.5])
accel['accelerate']  = fuzz.trapmf(accel.universe, [0.3, 1.5, 3, 3])

# === 3. Правила ===
rules = [
    ctrl.Rule(error['too_close'] & delta['approaching'], accel['brake']),
    ctrl.Rule(error['too_close'] & delta['steady'], accel['brake']),
    ctrl.Rule(error['too_close'] & delta['moving_away'], accel['hold']),
    ctrl.Rule(error['normal'] & delta['approaching'], accel['brake']),
    ctrl.Rule(error['normal'] & delta['steady'], accel['hold']),
    ctrl.Rule(error['normal'] & delta['moving_away'], accel['accelerate']),
    ctrl.Rule(error['far'] & delta['approaching'], accel['hold']),
    ctrl.Rule(error['far'] & delta['steady'], accel['accelerate']),
    ctrl.Rule(error['far'] & delta['moving_away'], accel['accelerate']),
]

accel_ctrl = ctrl.ControlSystem(rules)

for rule in accel_ctrl.rules:
    rule.activation = np.multiply

accel_sim = ctrl.ControlSystemSimulation(accel_ctrl)

# === 4. Параметры симуляции ===
dt = 0.1
T = 60.0
steps = int(T / dt)

d_ref = 25.0    # желаемое расстояние между авто (м)
v_auto = 15.0    # начальная скорость авто (м/с)
distance = 15.0 # начальная дистанция (м)

integral_error = 0.0
k_i = 100       
i_term_min = -1.5
i_term_max = 1.5

def v_lead_at_time(t):
    if t < 20.0:
        return 20.0
    elif t < 40.0:
        return 15.0
    else:
        return 22.0

# История
time_hist, dist_hist, v_auto_hist, v_lead_hist = [], [], [], []
accel_fuzzy_hist, accel_total_hist, error_hist, delta_hist, integral_hist = [], [], [], [], []

for i in range(steps):
    t = i * dt
    v_lead = v_lead_at_time(t)

    error_val = distance - d_ref
    delta_val = v_lead - v_auto

    accel_sim.input['error'] = error_val
    accel_sim.input['delta'] = delta_val
    accel_sim.compute()
    a_fuzzy = accel_sim.output['accel']

    integral_error += error_val * dt
    i_term = k_i * integral_error

    if i_term > i_term_max:
        i_term = i_term_max
        integral_error = i_term_max / k_i
    if i_term < i_term_min:
        i_term = i_term_min
        integral_error = i_term_min / k_i

    a_total = a_fuzzy + i_term
    a_total = max(-3.0, min(3.0, a_total))  # ограничение физическое

    v_auto += a_total * dt
    distance += (v_lead - v_auto) * dt

    time_hist.append(t)
    dist_hist.append(distance)
    v_auto_hist.append(v_auto)
    v_lead_hist.append(v_lead)
    accel_fuzzy_hist.append(a_fuzzy)
    accel_total_hist.append(a_total)
    error_hist.append(error_val)
    delta_hist.append(delta_val)
    integral_hist.append(i_term)

steady_start = int(45.0 / dt) 
steady_errors = np.array(error_hist[steady_start:])
print("Средняя ошибка: {:.3f} м".format(np.mean(steady_errors)))
print("Макс абс ошибка: {:.3f} м".format(np.max(np.abs(steady_errors))))

# === 7. Графики ===

# --- Окно 1: дистанция ---
plt.figure(figsize=(8, 5))
plt.plot(time_hist, dist_hist)
plt.axhline(d_ref, linestyle='--')
plt.ylabel('Дистанция (м)')
plt.xlabel('Время (с)')
plt.title('Изменение дистанции во времени')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Окно 2: скорости ---
plt.figure(figsize=(8, 5))
plt.plot(time_hist, v_auto_hist, label='Автопилот')
plt.plot(time_hist, v_lead_hist, label='Ведущее авто', linestyle='--')
plt.ylabel('Скорость (м/с)')
plt.xlabel('Время (с)')
plt.title('Скорости автомобилей во времени')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
