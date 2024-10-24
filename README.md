**Motor Speed Control via Hand Gesture and Python-Arduino Serial Communication**

This project demonstrates how to control the speed of a DC motor using hand gestures detected by Python, which are sent to an Arduino via serial communication. The Arduino receives gesture signals and speed values from Python, processes them, and adjusts the motor's speed and direction accordingly.

**01 Components Used**
+ Arduino Uno
+ DC motor
+ L298N Motor Driver
+ Python (with OpenCV and MediaPipe for gesture detection)
+ External 12V/9V power supply
+ Arduino Code
The Arduino code receives gestures and motor speed values via serial communication and adjusts the motor's speed accordingly. It supports the following features:

**02 Used Gestures**
LOVE YOU SIGNAL 🤟: Turn on the motor at a minimum speed
RAISE HAND 🤚     : Stop the motor.
THUMB IS DOWN 👎  : Decrease motor speed.
THUMB IS UP 👍    : Increase motor speed.
OKAY HAND 👌      : Set a specific motor speed.
RAISED FIST ✊    : Changing signal


**03 Pin Definitions & Connection**
+ ENA (Pin 6): Controls the motor speed (PWM).
+ IN1 (Pin 3): Controls the motor direction (input to the motor driver).
+ IN2 (Pin 5): Controls the motor direction (input to the motor driver).

************* Setup **************
The Arduino code initializes the motor pins and waits for serial communication. When a gesture is received, it adjusts the motor speed or stops the motor.

+ Connect the DC motor to the L298n motor driver. (Out 1, Out 2)
+ Connect the L298n motor driver to the Arduino (L298n ENA to Arduino PWM pin6, L298n IN1 and IN2 to Arduino PWM 3 and 5)
+ L298n GND to Arduino GND
+ L298n 5v to Arduino 5v
+ L298n 12v to powersupply (9/12v)


**04 How to Run the Project**

---------- Arduino ------------
+ Upload the provided Arduino code to the Arduino Uno.
+ Ensure that the motor driver and DC motor are connected to the Arduino.

----------- Python ------------
+ Install the required Python libraries:
pip install opencv-python mediapipe pyserial

+ Run the Python script for hand gesture detection:
Main.py

+ The Python script will detect gestures and send corresponding speed values to the Arduino.

------------- Expected Output ----------------
+ When Gesture 1 is detected, the motor will turn on and run at a low speed.
+ When Gesture 0 is detected, the motor will stop.
+ Gesture 2 and Gesture 3 will decrease and increase the motor's speed, respectively.
+ Gesture 4 allows for setting a specific speed for the motor.


**05 Additional Notes**
+ Ensure the correct COM port is used in the Python code for serial communication with the Arduino.
+ The motor speed value sent from Python should be between 0 and 255 (for PWM control).
