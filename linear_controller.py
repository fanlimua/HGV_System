import numpy as np

def linear_control(right_angle, left_angle_abs):
    """
    The simplest linear control: directly map the input angle to the output value
    Args:
        right_angle (float): The angle of the right thumb, range -90 to 90 degrees
        left_angle_abs (float): The absolute value of the left thumb angle, range 0 to 90 degrees
    
    Returns:
        tuple: (steering_signed, abs_speed)
    """
    right_angle = np.clip(right_angle, -90, 90)
    left_angle_abs = np.clip(left_angle_abs, 0, 90)
    
    steering_signed = right_angle / 2
    
    abs_speed = left_angle_abs * (6/90)
    
    return round(steering_signed, 2), round(abs_speed, 2)

def reset_linear():
    pass 