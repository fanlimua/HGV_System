**HGV System (Hand Gesture Vehicle System)**

This project implements a hand gesture-controlled vehicle system using fuzzy logic.  
The car's movement is controlled by hand angles captured from the user's gestures.


**Gesture Start & Stop Detection**

- **Start Detection**: Show a thumbs-up gesture to start gesture control.  
- **Stop Detection**: Open your palm fully facing the camera to stop gesture control.


**Right Hand – Steering**

- Turning the right hand to the right (positive angle) makes the car turn right.
- Turning the right hand to the left (negative angle) makes the car turn left.


**Left Hand – Speed**

- Holding the left hand vertically sets the car to the lowest speed.
- Tilting the left hand to either side increases the speed.


**fuzzy_controller.py**

- Implements the fuzzy control system  
- Input: Right and left hand angles  
- Output: Car's steering angle and speed

**linear_controller.py**
- Implements the linear control system  

**simulator.py**

- Draws the car and visualizes its movement on a 2D plane.

**Environment**

Installing Mediapipe 0.8.5 on jetson Nano
- We follow this link to install mediapipe-GPU on Jetson[https://jetson-docs.com/libraries/mediapipe/overview]
- To build the mediapipe-CPU
  - First, you need to install bazel in your system specifically 3.7.2 for 0.8.5  
  - Clone this mediapipe files [https://github.com/google-ai-edge/mediapipe]: ``` git clone -b 0.8.5 https://github.com/google-ai-edge/mediapipe.git```
  - Run the command: ``` cd mediapipe```
  -  Run this command to build the C++ program ```python3 setup.py install``` 
  - Then create the wheel ```python3 setup.py bdist_wheel```
  - Now you have folder named dist: ``` pip install dist/filename.whl```

**How to Run**

To launch the gesture-controlled **car simulator**, run:

```bash
python hand_control_sim.py --input camera # Using camera
python hand_control_sim.py --input video --video_path inputs/short_straight.mp4 --controller linear # Using pre-record video
```

To launch the gesture-controlled **remotely**, run:

```bash
python server_robot_receiver.py # Robot side
python client_camera_sender.py # Client side, need to modify the IP to server's IP
```

Evaluate **GPU efficiency** on the Jetson Nano: 
```bash
python3 hand_control_efficiency.py
```

Evaluate **CPU efficiency** on the Jetson Nano: 
```bash
python3 hand_control_CPU.py
```

**Demo Video**
[![Demo Video](https://drive.google.com/uc?export=view&id=1gkuTq4lmhjmGdQFhGYxvnMy5Y-Aq__7M)](https://drive.google.com/file/d/1Pbk-4f_Owc1YzzIoVLBz1tI2d2YnybkL/view?usp=sharing)
