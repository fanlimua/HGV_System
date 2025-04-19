import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define inputs
right_angle_input = ctrl.Antecedent(np.arange(-90, 91, 1), 'right_angle')
left_angle_input = ctrl.Antecedent(np.arange(0, 91, 1), 'left_angle')

# Define outputs
steering_output = ctrl.Consequent(np.arange(0, 46, 1), 'steering_angle')   # 0 to 45 degrees
speed_output = ctrl.Consequent(np.arange(0, 6.1, 0.1), 'speed')            # 0 to 5 units

# Membership functions (overlapping)
right_angle_input['strong_left'] = fuzz.trimf(right_angle_input.universe, [-90, -75, -60])
right_angle_input['left'] = fuzz.trimf(right_angle_input.universe, [-75, -45, -15])
right_angle_input['straight'] = fuzz.trimf(right_angle_input.universe, [-30, 0, 30])
right_angle_input['right'] = fuzz.trimf(right_angle_input.universe, [15, 45, 75])
right_angle_input['strong_right'] = fuzz.trimf(right_angle_input.universe, [60, 75, 90])

left_angle_input['very_low'] = fuzz.trimf(left_angle_input.universe, [0, 0, 20])
left_angle_input['low'] = fuzz.trimf(left_angle_input.universe, [10, 30, 50])
left_angle_input['medium'] = fuzz.trimf(left_angle_input.universe, [40, 55, 70])
left_angle_input['high'] = fuzz.trimf(left_angle_input.universe, [60, 75, 90])
left_angle_input['very_high'] = fuzz.trimf(left_angle_input.universe, [80, 90, 90])

steering_output['none'] = fuzz.trimf(steering_output.universe, [0, 0, 10])
steering_output['small'] = fuzz.trimf(steering_output.universe, [5, 15, 25])
steering_output['medium'] = fuzz.trimf(steering_output.universe, [20, 30, 40])
steering_output['large'] = fuzz.trimf(steering_output.universe, [35, 45, 45])

speed_output['stop'] = fuzz.trimf(speed_output.universe, [0, 0, 1])
speed_output['slow'] = fuzz.trimf(speed_output.universe, [0.5, 1.5, 2.5])
speed_output['medium'] = fuzz.trimf(speed_output.universe, [2, 3, 4])
speed_output['fast'] = fuzz.trimf(speed_output.universe, [3.5, 4.5, 5.5])
speed_output['very_fast'] = fuzz.trimf(speed_output.universe, [5, 6, 6])

# All combinations: 5x5 = 25 rules
rules = [
    ctrl.Rule(right_angle_input['strong_left'] & left_angle_input['very_low'], (steering_output['large'], speed_output['stop'])),
    ctrl.Rule(right_angle_input['strong_left'] & left_angle_input['low'],       (steering_output['large'], speed_output['slow'])),
    ctrl.Rule(right_angle_input['strong_left'] & left_angle_input['medium'],    (steering_output['large'], speed_output['medium'])),
    ctrl.Rule(right_angle_input['strong_left'] & left_angle_input['high'],      (steering_output['large'], speed_output['fast'])),
    ctrl.Rule(right_angle_input['strong_left'] & left_angle_input['very_high'], (steering_output['large'], speed_output['very_fast'])),

    ctrl.Rule(right_angle_input['left'] & left_angle_input['very_low'], (steering_output['medium'], speed_output['stop'])),
    ctrl.Rule(right_angle_input['left'] & left_angle_input['low'],       (steering_output['medium'], speed_output['slow'])),
    ctrl.Rule(right_angle_input['left'] & left_angle_input['medium'],    (steering_output['medium'], speed_output['medium'])),
    ctrl.Rule(right_angle_input['left'] & left_angle_input['high'],      (steering_output['medium'], speed_output['fast'])),
    ctrl.Rule(right_angle_input['left'] & left_angle_input['very_high'], (steering_output['medium'], speed_output['very_fast'])),

    ctrl.Rule(right_angle_input['straight'] & left_angle_input['very_low'], (steering_output['none'], speed_output['stop'])),
    ctrl.Rule(right_angle_input['straight'] & left_angle_input['low'],       (steering_output['none'], speed_output['slow'])),
    ctrl.Rule(right_angle_input['straight'] & left_angle_input['medium'],    (steering_output['none'], speed_output['medium'])),
    ctrl.Rule(right_angle_input['straight'] & left_angle_input['high'],      (steering_output['none'], speed_output['fast'])),
    ctrl.Rule(right_angle_input['straight'] & left_angle_input['very_high'], (steering_output['none'], speed_output['very_fast'])),

    ctrl.Rule(right_angle_input['right'] & left_angle_input['very_low'], (steering_output['medium'], speed_output['stop'])),
    ctrl.Rule(right_angle_input['right'] & left_angle_input['low'],       (steering_output['medium'], speed_output['slow'])),
    ctrl.Rule(right_angle_input['right'] & left_angle_input['medium'],    (steering_output['medium'], speed_output['medium'])),
    ctrl.Rule(right_angle_input['right'] & left_angle_input['high'],      (steering_output['medium'], speed_output['fast'])),
    ctrl.Rule(right_angle_input['right'] & left_angle_input['very_high'], (steering_output['medium'], speed_output['very_fast'])),

    ctrl.Rule(right_angle_input['strong_right'] & left_angle_input['very_low'], (steering_output['large'], speed_output['stop'])),
    ctrl.Rule(right_angle_input['strong_right'] & left_angle_input['low'],       (steering_output['large'], speed_output['slow'])),
    ctrl.Rule(right_angle_input['strong_right'] & left_angle_input['medium'],    (steering_output['large'], speed_output['medium'])),
    ctrl.Rule(right_angle_input['strong_right'] & left_angle_input['high'],      (steering_output['large'], speed_output['fast'])),
    ctrl.Rule(right_angle_input['strong_right'] & left_angle_input['very_high'], (steering_output['large'], speed_output['very_fast']))
]

# Build system
control_system = ctrl.ControlSystem(rules)
fuzzy_sim = ctrl.ControlSystemSimulation(control_system)

# Final function
def fuzzy_control(right_angle, left_angle_abs):
    # Clip input to range
    right_angle = np.clip(right_angle, -89.99, 89.99)
    left_angle_abs = np.clip(left_angle_abs, 0.01, 89.99)

    fuzzy_sim.input['right_angle'] = right_angle
    fuzzy_sim.input['left_angle'] = left_angle_abs

    try:
        fuzzy_sim.compute()
    except Exception as e:
        print("Fuzzy compute error:", e)
        return 0, 0

    # Extract outputs
    # print(f"{fuzzy_sim.output['steering_angle']}")
    abs_steering = round(fuzzy_sim.output['steering_angle'], 2)
    abs_speed = round(fuzzy_sim.output['speed'], 2)

    # Add direction sign to steering
    steering_signed = np.sign(right_angle) * abs_steering
    return steering_signed, abs_speed
