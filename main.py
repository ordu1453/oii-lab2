import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- Определение нечёткой системы ---
distance_to_leader = ctrl.Antecedent(np.arange(-100, 101, 1), 'distance')
change_of_distance = ctrl.Antecedent(np.arange(-20, 21, 1), 'change of distance')
v_autopilot = ctrl.Consequent(np.arange(0, 140, 1), 'v_autopilot')

# Функции принадлежности расстояния (теперь учитывают отрицательные значения)
distance_to_leader['Behind']   = fuzz.trapmf(distance_to_leader.universe, [-100, -100, -30, -5])  # автопилот позади
distance_to_leader['Close']    = fuzz.trimf(distance_to_leader.universe, [-10, 0, 10])            # рядом
distance_to_leader['Ahead']    = fuzz.trapmf(distance_to_leader.universe, [5, 30, 100, 100])      # автопилот впереди

# --- Изменение расстояния ---
change_of_distance['Closing'] = fuzz.trapmf(change_of_distance.universe, [-2, -2, -0.5, 0])
change_of_distance['Stable']  = fuzz.trimf(change_of_distance.universe, [-0.2, 0, 0.2])
change_of_distance['Opening'] = fuzz.trapmf(change_of_distance.universe, [0, 0.5, 2, 2])

# Функции принадлежности скорости автопилота
v_autopilot['Slow']   = fuzz.trapmf(v_autopilot.universe, [0, 0, 40, 80])
v_autopilot['Medium'] = fuzz.trimf(v_autopilot.universe, [70, 90, 120])
v_autopilot['Fast']   = fuzz.trapmf(v_autopilot.universe, [100, 120, 140, 140])

# --- Правила ---
rule1 = ctrl.Rule(distance_to_leader['Behind'] & change_of_distance['Opening'], v_autopilot['Fast'])
rule2 = ctrl.Rule(distance_to_leader['Behind'] & change_of_distance['Stable'], v_autopilot['Fast'])
rule3 = ctrl.Rule(distance_to_leader['Behind'] & change_of_distance['Closing'], v_autopilot['Medium'])

rule4 = ctrl.Rule(distance_to_leader['Close'] & change_of_distance['Stable'], v_autopilot['Medium'])
rule5 = ctrl.Rule(distance_to_leader['Close'] & change_of_distance['Closing'], v_autopilot['Slow'])
rule6 = ctrl.Rule(distance_to_leader['Close'] & change_of_distance['Opening'], v_autopilot['Medium'])

rule7 = ctrl.Rule(distance_to_leader['Ahead'] & change_of_distance['Stable'], v_autopilot['Slow'])
rule8 = ctrl.Rule(distance_to_leader['Ahead'] & change_of_distance['Closing'], v_autopilot['Slow'])
rule9 = ctrl.Rule(distance_to_leader['Ahead'] & change_of_distance['Opening'], v_autopilot['Medium'])

consys = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9
])
consysSim = ctrl.ControlSystemSimulation(consys)

# --- Параметры моделирования ---
dt = 0.1
time = 10   # секунд
steps = int(time / dt)

x_l = 20
v_l = 40     # скорость лидера
x_a = 0
v_a = 0
s = x_l - x_a
last_s = s

# --- Списки для сохранения ---
t_list = []
x_a_list = []
x_l_list = []
v_a_list = []
s_list = []

for i in range(steps):
    t = i * dt

    # обновляем координаты
    x_l += v_l * dt
    x_a += v_a * dt

    last_s = s
    s = x_l - x_a  # теперь может быть отрицательным
    ds = s - last_s  # изменение расстояния (может быть отрицательным)

    # нечёткий контроллер
    consysSim.input['distance'] = s
    consysSim.input['change of distance'] = ds
    consysSim.compute()

    v_a = consysSim.output['v_autopilot']  # новая скорость автопилота

    # сохраняем данные
    t_list.append(t)
    x_a_list.append(x_a)
    x_l_list.append(x_l)
    v_a_list.append(v_a)
    s_list.append(s)

# --- Графики ---
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_list, x_a_list, label='x_a (автопилот)')
plt.plot(t_list, x_l_list, label='x_l (лидер)')
plt.ylabel('Координата')
plt.title('Изменение координат x_a и x_l')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_list, s_list, label='Расстояние (s = x_l - x_a)')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Время (с)')
plt.ylabel('Расстояние')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
