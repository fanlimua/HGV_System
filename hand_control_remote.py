import cv2
import mediapipe as mp
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from fuzzy_controller_55 import fuzzy_control
from pid_controller import pid_control, reset_pid
from linear_controller import linear_control
from simulator import CarSimulator
import pandas as pd
import os
from datetime import datetime

# --- Configuration (set these as needed) ---
controller = 'fuzzy'  # 'fuzzy', 'pid', or 'linear'
plot = False           # Set True to enable real-time plotting
save_excel = False     # Set True to save control data to Excel
filter_alpha = 1       # Filter coefficient (0-1)
save_video = False     # Set True to save merged video

# --- Initialization ---
sim = CarSimulator()
if save_excel and not os.path.exists('control_data'):
    os.makedirs('control_data')

if save_excel:
    time_records = []
    right_angle_records = []
    left_angle_records = []
    steering_records = []
    speed_records = []
    pos_x_records = []
    pos_y_records = []
    start_time = None

if plot:
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle(f'{controller.capitalize()} Controller Output')
    ax1.set_title('Steering Angle')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_xlabel('Time (s)')
    ax1.grid(True)
    ax2.set_title('Speed')
    ax2.set_ylabel('Speed (units)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True)
    time_data = []
    steering_data = []
    speed_data = []
    start_time = time.time()
    steering_line, = ax1.plot([], [], 'r-')
    speed_line, = ax2.plot([], [], 'b-')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show(block=False)
    plt.pause(0.1)

video_writer = None
video_out_path = None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

class AngleFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.filtered_right = None
        self.filtered_left = None
    def update(self, right_angle, left_angle):
        if self.filtered_right is None and right_angle is not None:
            self.filtered_right = right_angle
        if self.filtered_left is None and left_angle is not None:
            self.filtered_left = left_angle
        if right_angle is not None:
            self.filtered_right = self.alpha * right_angle + (1 - self.alpha) * self.filtered_right
        if left_angle is not None:
            self.filtered_left = self.alpha * left_angle + (1 - self.alpha) * self.filtered_left
        return self.filtered_right, self.filtered_left
angle_filter = AngleFilter(alpha=filter_alpha)

started = False

def calculate_relative_angle(hand_landmarks, image_width, image_height, is_right):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    x1, y1 = int(thumb_tip.x * image_width), int(thumb_tip.y * image_height)
    x2, y2 = int(thumb_ip.x * image_width), int(thumb_ip.y * image_height)
    dx = x1 - x2
    dy = y2 - y1
    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)
    return round(angle_deg, 2)

def is_hand_open(hand_landmarks, threshold=0.22):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    distance = math.dist([wrist.x, wrist.y], [middle_tip.x, middle_tip.y])
    return distance > threshold

def are_fingers_folded(hand_landmarks):
    folded_finger_count = 0
    finger_ids = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_MCP),
    ]
    for tip_id, pip_id, mcp_id in finger_ids:
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]
        mcp = hand_landmarks.landmark[mcp_id]
        dist_tip_mcp = math.dist([tip.x, tip.y], [mcp.x, mcp.y])
        dist_pip_mcp = math.dist([pip.x, pip.y], [mcp.x, mcp.y])
        if dist_tip_mcp < dist_pip_mcp:
            folded_finger_count += 1
    return folded_finger_count == 4

def process_remote_frame(image):
    global started, video_writer, video_out_path, start_time
    # ...existing code for processing a single frame, replacing cap.read()...
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    image_height, image_width = image.shape[:2]
    left_line_x = image_width // 4
    right_line_x = image_width * 3 // 4
    left_angle = None
    right_angle = None
    cv2.line(image, (left_line_x, 0), (left_line_x, image_height), (200, 200, 200), 2)
    cv2.line(image, (right_line_x, 0), (right_line_x, image_height), (200, 200, 200), 2)
    sim_img = None
    if results.multi_hand_landmarks and results.multi_handedness:
        hands_data = list(zip(results.multi_hand_landmarks, results.multi_handedness))
        for hand_landmarks, _ in hands_data:
            if are_fingers_folded(hand_landmarks):
                started = True
                if save_excel and start_time is None:
                    start_time = time.time()
                break
        if started:
            cv2.putText(image, "Start", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            for hand_landmarks, handedness in hands_data:
                label = handedness.classification[0].label
                angle = calculate_relative_angle(hand_landmarks, image_width, image_height, is_right=(label == 'Right'))
                if label == 'Right':
                    right_angle = angle
                else:
                    left_angle = angle
                if right_angle is not None or left_angle is not None:
                    filtered_right, filtered_left = angle_filter.update(right_angle, left_angle)
                if right_angle is not None and left_angle is not None:
                    if controller == 'fuzzy':
                        steering_angle, speed = fuzzy_control(filtered_right, abs(filtered_left))
                    elif controller == 'pid':
                        steering_angle, speed = pid_control(filtered_right, abs(filtered_left))
                    else:
                        steering_angle, speed = linear_control(filtered_right, abs(filtered_left))
                    if save_excel and start_time is not None:
                        current_time = time.time() - start_time
                        time_records.append(current_time)
                        right_angle_records.append(right_angle)
                        left_angle_records.append(left_angle)
                        steering_records.append(steering_angle)
                        speed_records.append(speed)
                        car_pos_x, car_pos_y = sim.get_position()
                        pos_x_records.append(car_pos_x)
                        pos_y_records.append(car_pos_y)
                    if plot:
                        current_time = time.time() - start_time
                        time_data.append(current_time)
                        steering_data.append(steering_angle)
                        speed_data.append(speed)
                        max_points = 100000
                        if len(time_data) > max_points:
                            time_data = time_data[-max_points:]
                            steering_data = steering_data[-max_points:]
                            speed_data = speed_data[-max_points:]
                        steering_line.set_data(time_data, steering_data)
                        speed_line.set_data(time_data, speed_data)
                        ax1.relim()
                        ax1.autoscale_view()
                        ax2.relim()
                        ax2.autoscale_view()
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                    sim.update(steering_angle, speed)
                    sim_img = sim.draw()
                    cv2.putText(sim_img, f"Steering: {steering_angle:+.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2)
                    cv2.putText(sim_img, f"Speed: {speed:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2)
                    cv2.putText(sim_img, f"Controller: {controller}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2)
                    if save_excel:
                        cv2.putText(sim_img, "Recording Data", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(sim_img, f"Filter Alpha: {filter_alpha:.2f}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2)
                    cv2.imshow("Simulator", sim_img)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if is_hand_open(hand_landmarks):
                    started = False
                    if controller == 'pid':
                        reset_pid()
    if right_angle is not None:
        filtered_right, _ = angle_filter.update(None, None)
        cv2.putText(image, f"Right Angle: Raw={right_angle:+.1f} Filtered={filtered_right:+.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)
    if left_angle is not None:
        _, filtered_left = angle_filter.update(None, None)
        cv2.putText(image, f"Left Angle: Raw={left_angle:+.1f} Filtered={filtered_left:+.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
    cv2.imshow("Gesture Control", image)
    if save_video and sim_img is not None:
        if sim_img.shape[0] != image.shape[0]:
            scale = image.shape[0] / sim_img.shape[0]
            sim_img = cv2.resize(sim_img, (int(sim_img.shape[1]*scale), image.shape[0]))
        if sim_img.shape[1] != image.shape[1]:
            diff = abs(sim_img.shape[1] - image.shape[1])
            if sim_img.shape[1] < image.shape[1]:
                sim_img = cv2.copyMakeBorder(sim_img, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
            else:
                image = cv2.copyMakeBorder(image, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
        merged = np.hstack((image, sim_img))
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_out_path = f"merged_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            video_writer = cv2.VideoWriter(video_out_path, fourcc, 20.0, (merged.shape[1], merged.shape[0]))
        video_writer.write(merged)
    if cv2.waitKey(5) & 0xFF == 27:
        return False  # Signal to stop
    return True  # Continue

# Optionally, add a cleanup function to save data and release resources

def cleanup():
    global video_writer, video_out_path
    if save_excel and len(time_records) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"control_data/{controller}_data_{timestamp}.xlsx"
        data = {
            'Time': time_records,
            'Right_Angle': right_angle_records,
            'Left_Angle': left_angle_records,
            'Steering': steering_records,
            'Speed': speed_records,
            'Position_X': pos_x_records,
            'Position_Y': pos_y_records
        }
        df = pd.DataFrame(data)
        df.to_excel(excel_path, index=False)
        print(f"Data saved: {excel_path}")
        plt.figure(figsize=(8, 6))
        plt.plot(pos_x_records, pos_y_records, 'b-', linewidth=2)
        plt.grid(True)
        plt.title('Car Trajectory')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.axis('equal')
        trajectory_path = f"control_data/{controller}_trajectory_{timestamp}.png"
        plt.savefig(trajectory_path)
        print(f"Trail Path saved: {trajectory_path}")
    if plot:
        plt.close('all')
    if save_video and video_writer is not None:
        video_writer.release()
        print(f"Merged video saved: {video_out_path}")
    cv2.destroyAllWindows()
