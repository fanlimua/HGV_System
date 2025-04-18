#!/usr/bin/env python

import cv2
import mediapipe as mp
import math
import numpy as np
import rospy
import argparse
from geometry_msgs.msg import Twist
from fuzzy_controller import fuzzy_control

def calculate_relative_angle(hand_landmarks, image_width, image_height, is_right):
    """
    Calculate angle between thumb vector and vertical line
    """
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
    """
    Check if hand is open based on distance from wrist to middle finger tip
    """
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hand Gesture Control System')
    parser.add_argument('--input', type=str, default='camera', choices=['camera', 'video'],
                        help='Input source: camera or video file')
    parser.add_argument('--video_path', type=str, default='', 
                        help='Path to video file (required when input=video)')
    args = parser.parse_args()
    # Initialize ROS node
    rospy.init_node('hand_gesture_control', anonymous=True)
    
    # Create publisher for robot control
    cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    
    # Get control parameters
    linear_limit = rospy.get_param('~linear_speed_limit', 1.0)  # m/s
    angular_limit = rospy.get_param('~angular_speed_limit', 5.0)  # rad/s
    
    # Fuzzy controller output ranges
    FUZZY_MAX_STEERING_DEG = 45.0  # Max steering angle in degrees from fuzzy controller
    FUZZY_MAX_SPEED = 6.0  # Max speed in m/s from fuzzy controller
    
    # Initialize MediaPipe
    global mp_hands
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
    
    # Set the loop rate (10 Hz)
    rate = rospy.Rate(10)
    
    rospy.loginfo("Hand gesture control node started. Press 'Esc' to quit.")
    rospy.loginfo(f"Linear speed limit: {linear_limit} m/s, Angular speed limit: {angular_limit} rad/s")
    rospy.loginfo(f"Fuzzy control ranges - Steering: 0-{FUZZY_MAX_STEERING_DEG} deg, Speed: 0-{FUZZY_MAX_SPEED} m/s")

    while not rospy.is_shutdown() and cap.isOpened():
        success, image = cap.read()
        if not success:
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
                    
                    # Process both hands' angles to control the robot
                    if right_angle is not None and left_angle is not None:
                        # Use fuzzy control to get steering angle and speed
                        steering_angle_deg, fuzzy_speed = fuzzy_control(right_angle, abs(left_angle))
                        
                        # Convert steering angle from degrees to radians
                        # And normalize to the robot's angular velocity limit
                        # Note: fuzzy outputs 0-45 degrees, convert to -angular_limit to +angular_limit
                        steering_angle_rad = math.radians(steering_angle_deg)
                        
                        # Map the steering angle to the robot's range
                        # If FUZZY_MAX_STEERING_DEG is 45, then we map 0-45 degrees to 0-angular_limit
                        angular_z = (steering_angle_rad / math.radians(FUZZY_MAX_STEERING_DEG)) * angular_limit
                        
                        # Doudle check! Ensure angular velocity is within limits
                        angular_z = max(-angular_limit, min(angular_limit, angular_z))
                        
                        # Normalize speed from fuzzy controller range to robot's linear speed limit
                        linear_x = (fuzzy_speed / FUZZY_MAX_SPEED) * linear_limit
                        
                        # Doudle check! Ensure linear speed is within limits
                        linear_x = max(0, min(linear_limit, linear_x))
                        
                        # Set Twist message values
                        twist.linear.x = linear_x
                        twist.angular.z = angular_z
                        
                        # Display control values on screen
                        cv2.putText(image, f"Linear: {linear_x:.2f} m/s", (10, 160), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(image, f"Angular: {angular_z:.2f} rad/s", (10, 200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(image, f"Steering: {steering_angle_deg:.1f} deg", (10, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Log control values
                        rospy.loginfo(f"Linear X: {linear_x:.2f} m/s, Angular Z: {angular_z:.2f} rad/s, Steering: {steering_angle_deg:.1f} deg")

                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Check if hand is open to stop control
                    if is_hand_open(hand_landmarks):
                        started = False
                        # Stop the robot when hand opens
                        twist = Twist()  # Reset to zero
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
            cv2.putText(image, f"Right Thumb: {right_angle:+}", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)
        if left_angle is not None:
            cv2.putText(image, f"Left Thumb: {left_angle:+}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)

        cv2.imshow("Hand Gesture Control", image)
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