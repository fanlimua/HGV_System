#!/usr/bin/env python

import cv2
import mediapipe as mp
import math
import numpy as np
import rospy
import argparse
from geometry_msgs.msg import Twist
# Import all the controller options
from fuzzy_controller_35 import fuzzy_control
from pid_controller import pid_control, reset_pid
from linear_controller import linear_control

# Parse command line arguments
parser = argparse.ArgumentParser(description='Hand Gesture Control ROS Node')
parser.add_argument('--input', type=str, default='camera', choices=['camera', 'video'],
                    help='Input source: camera or video file')
parser.add_argument('--video_path', type=str, default='', 
                    help='Path to video file (required when input=video)')
parser.add_argument('--controller', type=str, default='fuzzy', choices=['fuzzy', 'pid', 'linear'],
                    help='Control system: fuzzy, pid, or linear')
parser.add_argument('--filter_alpha', type=float, default=0.5,
                    help='Filter coefficient (0-1): lower values = smoother control, higher values = more responsive')
args = parser.parse_args()

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
    return distance > threshold

def are_fingers_folded(hand_landmarks):
    """
    Check if index, middle, ring, and pinky fingers are folded.
    A finger is considered folded if:
        distance(tip, mcp) < distance(pip, mcp)
    """
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

        # Compute distances
        dist_tip_mcp = math.dist([tip.x, tip.y], [mcp.x, mcp.y])
        dist_pip_mcp = math.dist([pip.x, pip.y], [mcp.x, mcp.y])

        if dist_tip_mcp < dist_pip_mcp:
            folded_finger_count += 1

    return folded_finger_count == 4

# Implement the exponential moving average filter (EMA)
class AngleFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.filtered_right = None
        self.filtered_left = None
    
    def update(self, right_angle, left_angle):
        # When the filter is first called, initialize the filtered value
        if self.filtered_right is None and right_angle is not None:
            self.filtered_right = right_angle
        if self.filtered_left is None and left_angle is not None:
            self.filtered_left = left_angle
        
        # Use the EMA formula to update the filtered value
        if right_angle is not None:
            self.filtered_right = self.alpha * right_angle + (1 - self.alpha) * self.filtered_right
        if left_angle is not None:
            self.filtered_left = self.alpha * left_angle + (1 - self.alpha) * self.filtered_left
        
        return self.filtered_right, self.filtered_left

def main():
    # Initialize ROS node
    rospy.init_node('hand_gesture_control', anonymous=True)
    
    # Create publisher for robot control
    cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    
    # Get control parameters
    linear_limit = rospy.get_param('~linear_speed_limit', 1.0)  # m/s
    angular_limit = rospy.get_param('~angular_speed_limit', 5.0)  # rad/s
    
    # Controller output ranges
    MAX_STEERING_DEG = 45.0  # Max steering angle in degrees from controller
    MAX_SPEED = 6.0  # Max speed from controller

    # Initialize MediaPipe
    global mp_hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                          max_num_hands=2,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    # Create an instance of the angle filter
    angle_filter = AngleFilter(alpha=args.filter_alpha)
    
    # Start video capture
    if args.input == 'camera':
        cap = cv2.VideoCapture(0)
    else:
        if not args.video_path:
            rospy.logerr("Error: Video path must be provided when input=video")
            return
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            rospy.logerr(f"Error: Could not open video file {args.video_path}")
            return
    
    started = False
    
    # Set the loop rate (10 Hz)
    rate = rospy.Rate(10)
    
    rospy.loginfo("Hand gesture control node started. Press 'Esc' to quit.")
    rospy.loginfo(f"Using {args.controller} controller with filter alpha = {args.filter_alpha}")
    rospy.loginfo(f"Linear speed limit: {linear_limit} m/s, Angular speed limit: {angular_limit} rad/s")

    while not rospy.is_shutdown() and cap.isOpened():
        success, image = cap.read()
        if not success:
            # For video input, read failure means video end
            if args.input == 'video':
                rospy.loginfo("The video is finished playing.")
                break
            # For camera input, try to continue
            rospy.logwarn("Failed to capture frame from camera")
            continue

        image = cv2.flip(image, 1)  # Mirror flip for intuitive control
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

        # Create Twist message for robot control
        twist = Twist()

        if results.multi_hand_landmarks and results.multi_handedness:
            hands_data = list(zip(results.multi_hand_landmarks, results.multi_handedness))

            for hand_landmarks, _ in hands_data:
                if are_fingers_folded(hand_landmarks):  
                    started = True
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
                    
                    # Apply the filter to smooth the angle input
                    if right_angle is not None or left_angle is not None:
                        filtered_right, filtered_left = angle_filter.update(right_angle, left_angle)
                    
                    # Process both hands' angles to control the robot
                    if right_angle is not None and left_angle is not None:
                        # Use the selected controller
                        if args.controller == 'fuzzy':
                            steering_angle_deg, speed_value = fuzzy_control(filtered_right, abs(filtered_left))
                        elif args.controller == 'pid':
                            steering_angle_deg, speed_value = pid_control(filtered_right, abs(filtered_left))
                        else:  # linear
                            steering_angle_deg, speed_value = linear_control(filtered_right, abs(filtered_left))
                        
                        # Convert steering angle from degrees to radians
                        steering_angle_rad = math.radians(steering_angle_deg)
                        
                        # Map the steering angle to the robot's range
                        angular_z = (steering_angle_rad / math.radians(MAX_STEERING_DEG)) * angular_limit
                        
                        # Ensure angular velocity is within limits
                        angular_z = max(-angular_limit, min(angular_limit, angular_z))
                        
                        # Normalize speed from controller range to robot's linear speed limit
                        linear_x = (speed_value / MAX_SPEED) * linear_limit
                        
                        # Ensure linear speed is within limits
                        linear_x = max(0, min(linear_limit, linear_x))
                        
                        # Set Twist message values
                        twist.linear.x = linear_x
                        twist.angular.z = angular_z
                        
                        # Display control values on screen
                        cv2.putText(image, f"Linear: {linear_x:.2f} m/s", (10, 160), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(image, f"Angular: {angular_z:.2f} rad/s", (10, 200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(image, f"Steering: {steering_angle_deg:.1f} deg", (10, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Log control values
                        rospy.loginfo(f"Linear X: {linear_x:.2f} m/s, Angular Z: {angular_z:.2f} rad/s, Steering: {steering_angle_deg:.1f} deg")

                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Check if hand is open to stop control
                    if is_hand_open(hand_landmarks):
                        started = False
                        # Stop the robot when hand opens
                        twist = Twist()  # Reset to zero
                        if args.controller == 'pid':
                            reset_pid()  # Reset PID controller when stopping
            else:
                # Not started or hand opened - stop the robot
                twist = Twist()  # All values default to 0
        else:
            # No hands detected - stop the robot
            twist = Twist()
            
        # Publish command to robot
        cmd_vel_pub.publish(twist)

        # Display angles on screen
        if right_angle is not None:
            # Get the filtered value (only for display)
            filtered_right, _ = angle_filter.update(None, None)
            cv2.putText(image, f"Right Angle: Raw={right_angle:+.1f} Filtered={filtered_right:+.1f}", 
                      (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)
        if left_angle is not None:
            # Get the filtered value (only for display)
            _, filtered_left = angle_filter.update(None, None)
            cv2.putText(image, f"Left Angle: Raw={left_angle:+.1f} Filtered={filtered_left:+.1f}", 
                      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)

        cv2.imshow("Gesture Control", image)
        if cv2.waitKey(5) & 0xFF == 27:  # Esc key
            break
            
        rate.sleep()

    # Ensure robot stops when node is shutting down
    cmd_vel_pub.publish(Twist())
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass 