import numpy as np

def linear_control(right_angle, left_angle_abs):
    """
    最简单的线性控制：直接将输入角度映射到输出值
    
    Args:
        right_angle (float): 右手拇指角度，范围-90到90度
        left_angle_abs (float): 左手拇指角度绝对值，范围0到90度
    
    Returns:
        tuple: (steering_signed, abs_speed)
    """
    # 限制输入范围
    right_angle = np.clip(right_angle, -90, 90)
    left_angle_abs = np.clip(left_angle_abs, 0, 90)
    
    # 转向：将右手角度(-90到90)线性映射到转向角度(-45到45)
    # 简单地将输入除以2
    steering_signed = right_angle / 2
    
    # 速度：将左手角度(0到90)线性映射到速度(0到6)
    # 简单地将输入乘以6/90
    abs_speed = left_angle_abs * (6/90)
    
    return round(steering_signed, 2), round(abs_speed, 2)

# 不需要重置函数，因为这是无状态控制器
def reset_linear():
    # 无状态控制器，不需要重置
    pass 