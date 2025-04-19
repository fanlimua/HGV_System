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

- Implements the fuzzy inference system  
- Input: Right and left hand angles  
- Output: Car's steering angle and speed


**simulator.py**

- Draws the car and visualizes its movement on a 2D plane.


**How to Run**

To launch the gesture-controlled car simulator, run:

```bash
python hand_control.py
