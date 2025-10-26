import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

error = ctrl.Antecedent(np.arange(-10, 10.1, 0.1), 'error')
delta = ctrl.Antecedent(np.arange(-2, 2.1, 0.1), 'delta')
speed = ctrl.Consequent(np.arange(0, 20.1, 0.1), 'speed')

error['too_close'] = fuzz.trapmf(error.universe, [-10, -10, -6, -2])
error['normal'] = fuzz.trimf(error.universe, [-3, 0, 3])
error['far'] = fuzz.trapmf(error.universe, [2, 6, 10, 10])

delta['approaching'] = fuzz.trapmf(delta.universe, [-2, -2, -0.5, 0])
delta['steady'] = fuzz.trimf(delta.universe, [-0.3, 0, 0.3])
delta['moving_away'] = fuzz.trapmf(delta.universe, [0, 0.5, 2, 2])

speed['slow'] = fuzz.trapmf(speed.universe, [0, 0, 2, 6])
speed['medium'] = fuzz.trimf(speed.universe, [4, 8, 12])
speed['fast'] = fuzz.trapmf(speed.universe, [10, 14, 20, 20])

speed.view()

rules = [
    ctrl.Rule(error['too_close'] & delta['approaching'], speed['slow']),
    ctrl.Rule(error['too_close'] & delta['steady'], speed['slow']),
    ctrl.Rule(error['too_close'] & delta['moving_away'], speed['medium']),
    ctrl.Rule(error['normal'] & delta['approaching'], speed['slow']),
    ctrl.Rule(error['normal'] & delta['steady'], speed['medium']),
    ctrl.Rule(error['normal'] & delta['moving_away'], speed['fast']),
    ctrl.Rule(error['far'] & delta['approaching'], speed['medium']),
    ctrl.Rule(error['far'] & delta['steady'], speed['fast']),
    ctrl.Rule(error['far'] & delta['moving_away'], speed['fast']),
]

system = ctrl.ControlSystem(rules)

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
    # Лидер случайно меняет скорость
    # v_leader[i] = max(0, v_leader[i-1] + np.random.uniform(-0.1, 0.1))
    v_leader[i] = 5
    if time[i] > 30 and time[i] <60:
        v_leader[i] = 7
    elif time[i] > 60:
        v_leader[i] = 11 
    x_leader[i] = x_leader[i-1] + v_leader[i] * dt

    distance = x_leader[i] - x_auto[i-1]
    e = distance - desired_distance
    de = (e - error_list[i-1]) / dt if i > 1 else 0

    sim = ctrl.ControlSystemSimulation(system)
    sim.input['error'] = np.clip(e, -10, 10)
    sim.input['delta'] = np.clip(de, -2, 2)

    try:
        sim.compute()
        v_auto[i] = sim.output['speed']
    except KeyError:
        v_auto[i] = v_auto[i-1]

    x_auto[i] = x_auto[i-1] + v_auto[i] * dt
    error_list[i] = abs(e)

plt.figure(figsize=(12, 6))

# X-координаты
plt.subplot(2, 1, 1)
plt.plot(time, x_leader, label='Лидер', linewidth=2)
plt.plot(time, x_auto, label='Автопилот', linewidth=2)
plt.title('Координаты автомобилей')
plt.xlabel('Время, с')
plt.ylabel('X, м')
plt.legend()
plt.grid(True)

# Ошибка дистанции
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
