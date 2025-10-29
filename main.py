import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

distance_error = ctrl.Antecedent(np.arange(-10, 10.1, 0.1), 'distance_error')
delta_distance = ctrl.Antecedent(np.arange(-5, 5.1, 0.1), 'delta_distance')
v_follower = ctrl.Consequent(np.arange(0, 30.1, 0.1), 'v_follower', defuzzify_method="centroid")

distance_error['Negative'] = fuzz.trimf(distance_error.universe, [-10, -5, 0])
distance_error['Zero'] = fuzz.trimf(distance_error.universe, [-2, 0, 2])
distance_error['Positive'] = fuzz.trimf(distance_error.universe, [0, 5, 10])

# distance_error.view()

delta_distance['Negative'] = fuzz.trimf(delta_distance.universe, [-5, -2, 0])
delta_distance['Zero'] = fuzz.trimf(delta_distance.universe, [-1, 0, 1])
delta_distance['Positive'] = fuzz.trimf(delta_distance.universe, [0, 2, 5])

# delta_distance.view()

v_follower['Slow'] = fuzz.trimf(v_follower.universe, [0, 0, 12])
v_follower['Medium'] = fuzz.trimf(v_follower.universe, [8, 15, 22])
v_follower['Fast'] = fuzz.trimf(v_follower.universe, [18, 30, 30])

# v_follower.view()

rule1 = ctrl.Rule(distance_error['Positive'] & delta_distance['Negative'], v_follower['Slow'])
rule2 = ctrl.Rule(distance_error['Positive'] & delta_distance['Zero'], v_follower['Slow'])
rule3 = ctrl.Rule(distance_error['Positive'] & delta_distance['Positive'], v_follower['Medium'])

rule4 = ctrl.Rule(distance_error['Zero'] & delta_distance['Negative'], v_follower['Medium'])
rule5 = ctrl.Rule(distance_error['Zero'] & delta_distance['Zero'], v_follower['Medium'])
rule6 = ctrl.Rule(distance_error['Zero'] & delta_distance['Positive'], v_follower['Fast'])

rule7 = ctrl.Rule(distance_error['Negative'] & delta_distance['Negative'], v_follower['Medium'])
rule8 = ctrl.Rule(distance_error['Negative'] & delta_distance['Zero'], v_follower['Fast'])
rule9 = ctrl.Rule(distance_error['Negative'] & delta_distance['Positive'], v_follower['Fast'])

fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

dt = 0.1
t = np.arange(0, 100, dt)
d_ref = 5.0
v_leader = 13 + 3 * (t >= 110)

x_leader = np.zeros_like(t)
x_follower = np.zeros_like(t)
v_auto = np.zeros_like(t)

x_leader[0] = 0
x_follower[0] = -10
v_auto[0] = 0
error_int = 0
Ki = 0.5

for i in range(1, len(t)):
    x_leader[i] = x_leader[i-1] + v_leader[i-1] * dt
    x_follower[i] = x_follower[i-1] + v_auto[i-1] * dt

    distance = x_leader[i] - x_follower[i]
    error = distance - d_ref
    delta_error = (x_leader[i] - x_leader[i-1]) - (x_follower[i] - x_follower[i-1])

    error_int += error * dt
    e_fuzzy = error + Ki * error_int  

    fuzzy_sim.input['distance_error'] = e_fuzzy
    fuzzy_sim.input['delta_distance'] = delta_error
    fuzzy_sim.compute()

    v_auto[i] = 0.5 * v_auto[i-1] + 1.5* fuzzy_sim.output['v_follower']

# График ошибки по дистанции
plt.figure(figsize=(8, 5))
plt.plot(t, [x_leader[i] - x_follower[i] - d_ref for i in range(len(t))])
plt.title('График ошибки по дистанции')
plt.ylabel('Ошибка (м)')
plt.xlabel('Время (с)')
plt.grid()
plt.show()

# График изменения координат автомобилей
plt.figure(figsize=(8, 5))
plt.plot(t, x_leader, label='Лидер')
plt.plot(t, x_follower, label='Автопилот')
plt.title('График изменения координат автомобилей')
plt.ylabel('x (м)')
plt.xlabel('Время (с)')
plt.legend()
plt.grid()
plt.show()

# График изменения скорости автомобилей
plt.figure(figsize=(8, 5))
plt.plot(t, v_leader, label='Лидер')
plt.plot(t, v_auto, label='Автопилот')
plt.title('График изменения скорости автомобилей')
plt.ylabel('Скорость (м/с)')
plt.xlabel('Время (с)')
plt.legend()
plt.grid()
plt.show()

# # Расчет среднеквадратичной ошибки
# distance_errors = np.array([x_leader[i] - x_follower[i] - d_ref for i in range(len(t))])
# rmse = np.sqrt(np.mean(distance_errors**2))

# print(f"Среднеквадратичная ошибка (RMSE): {rmse:.4f} м")

# Расчет среднеквадратичной ошибки только для интервала времени 10–50 с
distance_errors = np.array([x_leader[i] - x_follower[i] - d_ref for i in range(len(t))])

# Индексы, где t находится в нужном диапазоне
mask = (t >= 5) & (t <= 50)

# Расчет RMSE только на выбранном промежутке
rmse_interval = np.sqrt(np.mean(distance_errors[mask]**2))

print(f"Среднеквадратичная ошибка (RMSE) на интервале 5–50 с: {rmse_interval:.4f} м")
