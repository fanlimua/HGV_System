import numpy as np
import cv2
import math

class CarSimulator:
    def __init__(self, width=600, height=600):
        self.canvas_size = (width, height)
        self.reset()

    def reset(self):
        # Initialize car position to center and reset trajectory
        self.position = np.array([self.canvas_size[0] // 2, self.canvas_size[1] // 2], dtype=np.float32)
        self.angle_deg = 0  # Orientation angle in degrees
        self.trajectory = []

    def get_position(self):
        """返回当前小车的(x, y)位置坐标"""
        return self.position[0], self.position[1]

    def update(self, steering_angle, speed):
        """
        steering_angle: rotation command (degrees), positive for right turn, negative for left
        speed: forward speed (e.g., 0–5 units)
        """
        self.angle_deg += steering_angle * 0.1  # Smooth turning by scaling steering effect
        angle_rad = math.radians(self.angle_deg)

        # Calculate movement vector
        dx = speed * math.cos(angle_rad)
        dy = speed * math.sin(angle_rad)

        # Update position
        self.position += np.array([dx, dy])
        self.trajectory.append(tuple(self.position.astype(int)))

        # Keep position within canvas bounds
        self.position[0] = np.clip(self.position[0], 0, self.canvas_size[0] - 1)
        self.position[1] = np.clip(self.position[1], 0, self.canvas_size[1] - 1)

    def draw(self):
        # Create a dark canvas (black background)
        canvas = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)

        # Optional: use dark gray instead of black
        # canvas[:] = (30, 30, 30)

        # Draw the trajectory path
        for point in self.trajectory:
            cv2.circle(canvas, point, 2, (255, 100, 100), -1)  # Light red trail points

        # Draw the car as a circle
        car_x, car_y = self.position.astype(int)
        cv2.circle(canvas, (car_x, car_y), 6, (0, 255, 255), -1)  # Yellow-green body

        # Draw the heading direction as an arrow
        end_x = int(car_x + 20 * math.cos(math.radians(self.angle_deg)))
        end_y = int(car_y + 20 * math.sin(math.radians(self.angle_deg)))
        cv2.arrowedLine(canvas, (car_x, car_y), (end_x, end_y), (255, 255, 255), 2, tipLength=0.4)  # White arrow

        return canvas
