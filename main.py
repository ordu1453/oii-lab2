import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# -------------------------------
# 1. Определяем нечёткие переменные
# -------------------------------
e = ctrl.Antecedent(np.arange(-10, 10.1, 0.1), 'e')   # ошибка расстояния
de = ctrl.Antecedent(np.arange(-5, 5.1, 0.1), 'de')   # изменение ошибки
v = ctrl.Consequent(np.arange(0, 101, 1), 'v')        # выход — скорость ведомого (%)

# Функции принадлежности (перекрывающиеся зоны)
e['NB'] = fuzz.trapmf(e.universe, [-10, -10, -6, -3])
e['NS'] = fuzz.trimf(e.universe, [-6, -3, 0])
e['Z']  = fuzz.trimf(e.universe, [-3, 0, 3])
e['PS'] = fuzz.trimf(e.universe, [0, 3, 6])
e['PB'] = fuzz.trapmf(e.universe, [3, 6, 10, 10])

de['NB'] = fuzz.trapmf(de.universe, [-5, -5, -3, -1.5])
de['NS'] = fuzz.trimf(de.universe, [-3, -1.5, 0])
de['Z']  = fuzz.trimf(de.universe, [-1.5, 0, 1.5])
de['PS'] = fuzz.trimf(de.universe, [0, 1.5, 3])
de['PB'] = fuzz.trapmf(de.universe, [1.5, 3, 5, 5])

v['VS'] = fuzz.trapmf(v.universe, [0, 0, 10, 20])
v['S']  = fuzz.trimf(v.universe, [10, 25, 40])
v['M']  = fuzz.trimf(v.universe, [30, 50, 70])
v['F']  = fuzz.trimf(v.universe, [60, 75, 90])
v['VF'] = fuzz.trapmf(v.universe, [80, 90, 100, 100])

# e.view()
# de.view()
# v.view()


# 2. Набор правил 
rules = [
    ctrl.Rule(e['NB'] & de['NB'], v['VS']),
    ctrl.Rule(e['NB'] & de['Z'], v['S']),
    ctrl.Rule(e['NB'] & de['PB'], v['M']),
    ctrl.Rule(e['Z']  & de['Z'], v['M']),
    ctrl.Rule(e['PB'] & de['NB'], v['F']),
    ctrl.Rule(e['PB'] & de['PB'], v['VF']),
    ctrl.Rule(e['PS'] & de['PS'], v['F']),
    ctrl.Rule(e['NS'] & de['NB'], v['S']),
    ctrl.Rule(e['NS'] & de['Z'], v['M']),
    ctrl.Rule(e['Z']  & de['PB'], v['F'])
]

rules[1].view()

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

sim.input['e'] = 3.5
sim.input['de'] = 6
sim.compute()

print(sim.output['v'])
e.view(sim = sim)
de.view(sim = sim)
v.view(sim = sim)
plt.show()



# # -------------------------------
# # 3. Параметры симуляции
# # -------------------------------
# dt = 0.2         # шаг по времени (с)
# T = 25           # длительность (с)
# steps = int(T / dt)

# D_ref = 5.0      # желаемая дистанция (м)
# D = 10.0         # начальная дистанция
# V_leader = 10.0  # скорость лидера (м/с)
# V_max = 25.0     # макс. скорость ведомого (м/с)
# V_follower = 5.0 # начальная скорость ведомого (м/с)

# distances, speeds = [], []
# times = np.arange(0, T, dt)
# prev_e = D - D_ref



# # -------------------------------
# # 4. Основной цикл
# # -------------------------------
# for t in times:
#     e_input = D - D_ref
#     de_input = e_input - prev_e

#     # создаём новый симулятор на каждом шаге (иначе ломается)
#     sim = ctrl.ControlSystemSimulation(system)
#     sim.input['e'] = np.clip(e_input, -10, 10)
#     sim.input['de'] = np.clip(de_input, -5, 5)

#     try:
#         sim.compute()
#         v_percent = sim.output.get('v', 50.0)
#     except Exception as err:
#         print(f"[{t:.1f}s] Ошибка вывода: {err}")
#         v_percent = 50.0

#     V_follower = (v_percent / 100) * V_max
#     D += (V_leader - V_follower) * dt
#     prev_e = e_input

#     distances.append(D)
#     speeds.append(V_follower)

# # -------------------------------
# # 5. Графики
# # -------------------------------
# plt.figure(figsize=(10,6))

# plt.subplot(2,1,1)
# plt.plot(times, distances, label='Дистанция D(t)', color='b')
# plt.axhline(D_ref, color='r', linestyle='--', label='Желаемая дистанция D_ref')
# plt.ylabel('Расстояние (м)')
# plt.legend()
# plt.grid(True)

# plt.subplot(2,1,2)
# plt.plot(times, speeds, label='Скорость ведомого', color='g')
# plt.axhline(V_leader, color='r', linestyle='--', label='Скорость лидера')
# plt.xlabel('Время (с)')
# plt.ylabel('Скорость (м/с)')
# plt.legend()
# plt.grid(True)

# plt.suptitle('Fuzzy Follow (алгоритм Ларсена, X=1)')
# plt.tight_layout()
# plt.show()

# # -------------------------------
# # 6. Оценка качества
# # -------------------------------
# error = np.array(distances) - D_ref
# rmse = np.sqrt(np.mean(error**2))
# print(f"\nСреднеквадратичная ошибка дистанции: {rmse:.3f} м")
