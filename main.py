import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# -------------------------------
# 1. Определяем нечёткие переменные
# -------------------------------
e = ctrl.Antecedent(np.arange(-10, 10.1, 0.1), 'e')   
de = ctrl.Antecedent(np.arange(-5, 5.1, 0.1), 'de')   
v = ctrl.Consequent(np.arange(0, 101, 1), 'v')        

# Функции принадлежности
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

e.view()
de.view()
# 2. Набор правил (Ларсен)

# Для Ларсена вместо min используем "product" при вычислении активации
# В skfuzzy нет прямого параметра, поэтому используем метод defuzzify_method='centroid' и product вручную

def larsen_rule(firing_strength, consequent_mf):
    # Умножаем всю функцию принадлежности на силу активации (product)
    return firing_strength * consequent_mf

# Создаём "систему вручную"
rules = [
    (('NB', 'NB'), 'VS'),
    (('NB', 'Z'), 'S'),
    (('NB', 'PB'), 'M'),
    (('Z',  'Z'), 'M'),
    (('PB', 'NB'), 'F'),
    (('PB', 'PB'), 'VF'),
    (('PS', 'PS'), 'F'),
    (('NS', 'NB'), 'S'),
    (('NS', 'Z'), 'M'),
    (('Z',  'PB'), 'F')
]

# Пример вычисления для конкретного входа
e_input = 2.5
de_input = 4.5

# Фаззификация
e_level = {key: fuzz.interp_membership(e.universe, e[key].mf, e_input) for key in e.terms.keys()}
de_level = {key: fuzz.interp_membership(de.universe, de[key].mf, de_input) for key in de.terms.keys()}

print(e_level)
print(de_level)

# Агрегация (Ларсен)
output_aggregated = np.zeros_like(v.universe)
for (e_term, de_term), v_term in rules:
    firing_strength = e_level[e_term] * de_level[de_term]  # product вместо min
    output_aggregated = np.fmax(output_aggregated, larsen_rule(firing_strength, v[v_term].mf))

print(firing_strength)
print(output_aggregated)

# Дефаззификация
v_output = fuzz.defuzz(v.universe, output_aggregated, 'centroid')
print("Скорость ведомого:", v_output)

# Визуализация
plt.figure()
plt.plot(v.universe, output_aggregated, 'r', linewidth=2)
v.view()
plt.show()
