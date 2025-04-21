import cv2
import mediapipe as mp
import math
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from fuzzy_controller import fuzzy_control
from fuzzy_controller_35 import fuzzy_control
from pid_controller import pid_control, reset_pid
from linear_controller import linear_control
from simulator import CarSimulator
import pandas as pd
import os
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='Hand Gesture Control System')
parser.add_argument('--input', type=str, default='camera', choices=['camera', 'video'],
                    help='Input source: camera or video file')
parser.add_argument('--video_path', type=str, default='', 
                    help='Path to video file (required when input=video)')
parser.add_argument('--controller', type=str, default='fuzzy', choices=['fuzzy', 'pid', 'linear'],
                    help='Control system: fuzzy, pid, or linear')
parser.add_argument('--plot', action='store_true', 
                    help='Enable real-time plotting of controller outputs')
parser.add_argument('--save_excel', action='store_true',
                    help='Save control data to Excel file')
parser.add_argument('--filter_alpha', type=float, default=1,
                    help='Filter coefficient (0-1): lower values = smoother control, higher values = more responsive')
args = parser.parse_args()

# create the directory to save the data
if args.save_excel and not os.path.exists('control_data'):
    os.makedirs('control_data')

sim = CarSimulator()

# list to store the control data
if args.save_excel:
    time_records = []
    right_angle_records = []
    left_angle_records = []
    steering_records = []
    speed_records = []
    # add the position record list
    pos_x_records = []
    pos_y_records = []
    start_time = None

# list to store the controller output data
if args.plot:
    # create the canvas and subplots
    plt.ion()  # interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle(f'{args.controller.capitalize()} Controller Output')
    
    # set the subplot titles and labels
    ax1.set_title('Steering Angle')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_xlabel('Time (s)')
    ax1.grid(True)
    
    ax2.set_title('Speed')
    ax2.set_ylabel('Speed (units)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True)
    
    # initialize the data lists
    time_data = []
    steering_data = []
    speed_data = []
    
    # record the start time
    start_time = time.time()
    
    # initialize the lines
    steering_line, = ax1.plot([], [], 'r-')
    speed_line, = ax2.plot([], [], 'b-')
    
    # set the display window
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show(block=False)
    plt.pause(0.1)

# Calculate angle between thumb vector and vertical line
def calculate_relative_angle(hand_landmarks, image_width, image_height, is_right):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

    # Convert to pixel coordinates
    x1, y1 = int(thumb_tip.x * image_width), int(thumb_tip.y * image_height)
    x2, y2 = int(thumb_ip.x * image_width), int(thumb_ip.y * image_height)

    # Thumb vector
    dx = x1 - x2
    dy = y2 - y1  # invert y for standard math coord

    # Reference vertical line direction: (0, 1)
    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)

    return round(angle_deg, 2)

def is_hand_open(hand_landmarks, threshold=0.22):

    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    
    distance = math.dist([wrist.x, wrist.y], [middle_tip.x, middle_tip.y])
    # print(f"Distance: {distance}")

    return distance > threshold

def are_fingers_folded(hand_landmarks):
    """
    Check if index, middle, ring, and pinky fingers are folded.
    A finger is considered folded if:
        distance(tip, mcp) < distance(pip, mcp)
    """
    folded_finger_count = 0

    finger_ids = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_MCP, "Index Finger"),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, "Middle Finger"),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_MCP, "Ring Finger"),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_MCP, "Pinky Finger"),
    ]

    # thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    # thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

    # if thumb_tip.y < thumb_ip.y:
    #     folded_finger_count += 1

    for tip_id, pip_id, mcp_id, finger_name in finger_ids:
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]
        mcp = hand_landmarks.landmark[mcp_id]

        # Compute distances
        dist_tip_mcp = math.dist([tip.x, tip.y], [mcp.x, mcp.y])
        dist_pip_mcp = math.dist([pip.x, pip.y], [mcp.x, mcp.y])
        # print(f"Tip: {[tip.x, tip.y]}; Pip: {[pip.x, pip.y]}; Mcp: {[mcp.x, mcp.y]}.")

        if dist_tip_mcp < dist_pip_mcp:
            folded_finger_count += 1
        # if dist_tip_mcp < dist_pip_mcp:
        #     print(f"{finger_name}: Tip to MCP distance < Pip to MCP distance")
        #     folded_finger_count += 1
        # else:
        #     print(f"{finger_name}: Tip to MCP distance >= Pip to MCP distance")

    return folded_finger_count == 4

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
if args.input == 'camera':
    cap = cv2.VideoCapture(0)
else:
    if not args.video_path:
        print("Error: Video path must be provided when input=video")
        exit(1)
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        exit(1)

started = False

# implement the exponential moving average filter (EMA)
class AngleFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.filtered_right = None
        self.filtered_left = None
    
    def update(self, right_angle, left_angle):
        # when the filter is first called, initialize the filtered value
        if self.filtered_right is None and right_angle is not None:
            self.filtered_right = right_angle
        if self.filtered_left is None and left_angle is not None:
            self.filtered_left = left_angle
        
        # use the EMA formula to update the filtered value
        if right_angle is not None:
            self.filtered_right = self.alpha * right_angle + (1 - self.alpha) * self.filtered_right
        if left_angle is not None:
            self.filtered_left = self.alpha * left_angle + (1 - self.alpha) * self.filtered_left
        
        return self.filtered_right, self.filtered_left

# create an instance of the angle filter
angle_filter = AngleFilter(alpha=args.filter_alpha)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        # for video input, read failure means video end
        if args.input == 'video':
            print("The video is finished playing and the program exits normally")
            break
        # for camera input, try to continue
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    image_height, image_width = image.shape[:2]
    left_line_x = image_width // 4
    right_line_x = image_width * 3 // 4

    left_angle = None
    right_angle = None

    # Draw vertical reference lines
    cv2.line(image, (left_line_x, 0), (left_line_x, image_height), (200, 200, 200), 2)
    cv2.line(image, (right_line_x, 0), (right_line_x, image_height), (200, 200, 200), 2)

    if results.multi_hand_landmarks and results.multi_handedness:
        hands_data = list(zip(results.multi_hand_landmarks, results.multi_handedness))

        for hand_landmarks, _ in hands_data:
            if are_fingers_folded(hand_landmarks):  
                started = True
                # record the start time (if data saving is enabled and not started yet)
                if args.save_excel and start_time is None:
                    start_time = time.time()
                break  

        if started:
            cv2.putText(image, "Start", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            for hand_landmarks, handedness in hands_data:
                label = handedness.classification[0].label  # 'Left' or 'Right' Hand

                angle = calculate_relative_angle(
                    hand_landmarks, image_width, image_height, is_right=(label == 'Right')
                )
                if label == 'Right':
                    right_angle = angle
                else:
                    left_angle = angle
                
                # print(f"{right_angle}, {left_angle}")
                
                # apply the filter to smooth the angle input
                if right_angle is not None or left_angle is not None:
                    filtered_right, filtered_left = angle_filter.update(right_angle, left_angle)
                    
                    # if you need to debug, you can print the original value and the filtered value
                    # print(f"Raw: {right_angle}, {left_angle} | Filtered: {filtered_right:.2f}, {filtered_left:.2f}")
                
                if right_angle is not None and left_angle is not None:
                    # use the filtered angle value for control
                    if args.controller == 'fuzzy':
                        steering_angle, speed = fuzzy_control(filtered_right, abs(filtered_left))
                    elif args.controller == 'pid':
                        steering_angle, speed = pid_control(filtered_right, abs(filtered_left))
                    else:  # linear
                        steering_angle, speed = linear_control(filtered_right, abs(filtered_left))
                    
                    # save data to list - record the original angle and the filtered angle
                    if args.save_excel and start_time is not None:
                        current_time = time.time() - start_time
                        time_records.append(current_time)
                        right_angle_records.append(right_angle)  # original angle
                        left_angle_records.append(left_angle)    # original angle
                        steering_records.append(steering_angle)
                        speed_records.append(speed)
                        
                        # get the car current position
                        car_pos_x, car_pos_y = sim.get_position()
                        pos_x_records.append(car_pos_x)
                        pos_y_records.append(car_pos_y)
                    
                    # update the real-time chart
                    if args.plot:
                        current_time = time.time() - start_time
                        time_data.append(current_time)
                        steering_data.append(steering_angle)
                        speed_data.append(speed)
                        
                        # limit the number of data points, keep the latest 100 points
                        max_points = 100000
                        if len(time_data) > max_points:
                            time_data = time_data[-max_points:]
                            steering_data = steering_data[-max_points:]
                            speed_data = speed_data[-max_points:]
                        
                        # update the chart data
                        steering_line.set_data(time_data, steering_data)
                        speed_line.set_data(time_data, speed_data)
                        
                        # auto adjust the axis range
                        ax1.relim()
                        ax1.autoscale_view()
                        ax2.relim()
                        ax2.autoscale_view()
                        
                        # refresh the chart
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        
                    sim.update(steering_angle, speed)
                    sim_img = sim.draw()
                    
                    cv2.putText(sim_img, f"Steering: {steering_angle:+.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2)
                    cv2.putText(sim_img, f"Speed: {speed:.2f}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2)
                    cv2.putText(sim_img, f"Controller: {args.controller}", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2)
                    
                    # show the recording data
                    if args.save_excel:
                        cv2.putText(sim_img, "Recording Data", (10, 115),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # show the filter information
                    cv2.putText(sim_img, f"Filter Alpha: {args.filter_alpha:.2f}", (10, 145),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2)
                                   
                    cv2.imshow("Simulator", sim_img)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if is_hand_open(hand_landmarks):
                    started = False
                    if args.controller == 'pid':
                        reset_pid()  # Reset PID controller when stopping

    # Display angles
    if right_angle is not None:
        # get the filtered value (only for display)
        filtered_right, _ = angle_filter.update(None, None)
        cv2.putText(image, f"Right Angle: Raw={right_angle:+.1f} Filtered={filtered_right:+.1f}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)
    if left_angle is not None:
        # get the filtered value (only for display)
        _, filtered_left = angle_filter.update(None, None)
        cv2.putText(image, f"Left Angle: Raw={left_angle:+.1f} Filtered={filtered_left:+.1f}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
    
    cv2.imshow("Gesture Control", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# if data saving is enabled and there is data, save to Excel
if args.save_excel and len(time_records) > 0:
    # create a file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = f"control_data/{args.controller}_data_{timestamp}.xlsx"
    
    # Save data to Excel
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
    
    # visualize the trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(pos_x_records, pos_y_records, 'b-', linewidth=2)
    plt.grid(True)
    plt.title('Car Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')  
    trajectory_path = f"control_data/{args.controller}_trajectory_{timestamp}.png"
    plt.savefig(trajectory_path)
    print(f"Trail Path saved: {trajectory_path}")

cap.release()
cv2.destroyAllWindows()
if args.plot:
    plt.close(fig)
