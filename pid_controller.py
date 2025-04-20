import numpy as np
import time

class PIDController:
    def __init__(self, kp, ki, kd):
        # PID coefficients
        self.kp = kp
        self.ki = ki 
        self.kd = kd
        
        # Error tracking
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        
    def compute(self, setpoint, current_value):
        # Calculate time delta
        current_time = time.time()
        dt = current_time - self.last_time
        dt = max(dt, 0.001)  # Prevent division by zero
        
        # Calculate error
        error = setpoint - current_value
        
        # Calculate PID terms
        p_term = self.kp * error
        
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Store values for next iteration
        self.prev_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

# Initialize PID controllers - tune these parameters as needed
steering_pid = PIDController(kp=0.3, ki=0.1, kd=0.05)
speed_pid = PIDController(kp=0.3, ki=0.1, kd=0.05)

def pid_control(right_angle, left_angle_abs):
    """
    PID control for steering and speed
    
    Args:
        right_angle (float): Angle from -90 to 90 degrees
        left_angle_abs (float): Absolute angle from 0 to 90 degrees
    
    Returns:
        tuple: (steering_signed, abs_speed)
    """
    # Clip input to range
    right_angle = np.clip(right_angle, -90, 90)
    left_angle_abs = np.clip(left_angle_abs, 0, 90)
    
    # Steering control
    # Target angle is 0 (straight)
    steering_target = 0
    steering_output = steering_pid.compute(steering_target, right_angle)
    
    # Convert PID output to expected steering range (0-45 degrees)
    # and add direction sign based on right_angle
    abs_steering = min(abs(steering_output), 45)
    steering_signed = np.sign(right_angle) * abs_steering
    
    # Speed control
    # Map left_angle_abs (0-90) to speed setpoint (0-6)
    # Higher angle = higher speed
    speed_target = left_angle_abs / 90 * 6
    
    # Current speed is assumed to be 0, so error is just the target
    # This is a simplified approach - in a real system you'd use actual speed feedback
    speed_output = speed_pid.compute(speed_target, 0)
    
    # Ensure speed is within expected range (0-6)
    abs_speed = np.clip(speed_output, 0, 6.0)
    
    return round(steering_signed, 2), round(abs_speed, 2)

# Reset function to clear PID state
def reset_pid():
    steering_pid.reset()
    speed_pid.reset() 