import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


distance_to_leader = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')
change_of_distance = ctrl.Antecedent(np.arange(0, 21, 1), 'change of distance')
v_autopilot = ctrl.Consequent(np.arange(0, 101, 1), 'v_autopilot')

distance_to_leader['TooClose'] = fuzz.trapmf(distance_to_leader.universe, [0,0,10,30])
distance_to_leader['Safe']     = fuzz.trimf(distance_to_leader.universe, [20,50,80])
distance_to_leader['Far']      = fuzz.trapmf(distance_to_leader.universe, [70,90,100,100])

# distance_to_leader.view()

# Функции принадлежности скорости автопилота
v_autopilot['Slow']   = fuzz.trapmf(v_autopilot.universe, [0,0,20,40])
v_autopilot['Medium'] = fuzz.trimf(v_autopilot.universe, [30,50,70])
v_autopilot['Fast']   = fuzz.trapmf(v_autopilot.universe, [60,80,100,100])

# v_autopilot.view()

# Функции принадлежности скорости лидера
change_of_distance['Zero']   = fuzz.trapmf(change_of_distance.universe, [0,0,0,10])
change_of_distance['Medium'] = fuzz.trimf(change_of_distance.universe, [3,10,17])
change_of_distance['Fast']   = fuzz.trapmf(change_of_distance.universe, [10,20,20,20])

# change_of_distance.view()

# plt.show()

# Example
# rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
# rule2 = ctrl.Rule(service['average'], tip['medium'])
# rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])

# rule1.view()

rule1 =ctrl.Rule(distance_to_leader['Far'] & change_of_distance['Zero'], v_autopilot['Fast'])
rule2 =ctrl.Rule(distance_to_leader['Far'] & change_of_distance['Medium'], v_autopilot['Fast'])
rule3 =ctrl.Rule(distance_to_leader['Far'] & change_of_distance['Fast'], v_autopilot['Medium'])
rule4 =ctrl.Rule(distance_to_leader['Safe'] & change_of_distance['Zero'], v_autopilot['Medium'])
rule5 =ctrl.Rule(distance_to_leader['Safe'] & change_of_distance['Medium'], v_autopilot['Slow'])
rule6 =ctrl.Rule(distance_to_leader['Safe'] & change_of_distance['Fast'], v_autopilot['Slow'])
rule7 =ctrl.Rule(distance_to_leader['TooClose'] & change_of_distance['Zero'], v_autopilot['Slow'])
rule8 =ctrl.Rule(distance_to_leader['TooClose'] & change_of_distance['Medium'], v_autopilot['Slow'])
rule9 =ctrl.Rule(distance_to_leader['TooClose'] & change_of_distance['Fast'], v_autopilot['Slow'])

# Example
# tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

consys = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

# Example
# tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

consysSim = ctrl.ControlSystemSimulation(consys)

# Example
# # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
# tipping.input['quality'] = 6.5
# tipping.input['service'] = 9.8

consysSim.input['distance'] = 20
consysSim.input['change of distance'] = 10

# # Crunch the numbers
# tipping.compute()

consysSim.compute()

# Example
# print tipping.output['tip']
# tip.view(sim=tipping)

print(consysSim.output['v_autopilot'])

v_autopilot.view(sim=consysSim)

plt.show()




