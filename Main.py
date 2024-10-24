import cv2
import mediapipe as mp
import serial
import time
import math

cap = cv2.VideoCapture(0)
speed = 135
onSpeed = 135
pTime = 0
cTime = 0
last_gesture = "none"
current_gesture = ""
signal = ''

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

try:
    arduino = serial.Serial(port='COM6', baudrate=9600, timeout=1, write_timeout=2)
    time.sleep(2)  # Wait for the serial connection to initialize
    arduino_connected = True

    # Send data to Arduino
    arduino.write(b'1')  # Send '1' to the Arduino
    print("=========================================")
    print("Sent request to arduino")
    # Wait a moment for the Arduino to respond
    time.sleep(1)

    # Read response from Arduino
    response = arduino.readline().decode('utf-8').strip()  # Read and decode response
    if response:
        print("Arduino response to the request")
    else:
        print("No response from Arduino")

except serial.serialutil.SerialException as e:
    print(f"Error Connecting to Arduino: {e}")
    arduino_connected = False

print("Arduino connection is : ", arduino_connected)
print("=========================================")

################################### SAMU ############################################
# FUNCTION TO CALCULATE ANGLES OF FINGER
def calculate_angle(a, b, c):
    """Calculate the angle (in degrees) between three points (a, b, c)."""
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    # Set the cosine value in range in -1 to 1
    cosine_angle = max(-1, min(1, cosine_angle))
    try:
        angle = math.degrees(math.acos(cosine_angle))
    except ValueError:
        angle = 0
    return angle

################################### SAMU ############################################
# FUNCTION FOR FINDING IF FINGERS ARE CURLED OR NOT
def is_finger_curled(mcp, pip, dip, tip):
    """Determine if a finger is curled based on the angles at its joints."""
    angle_pip = calculate_angle(mcp, pip, dip)
    angle_dip = calculate_angle(pip, dip, tip)
    return angle_pip < 160 or angle_dip < 160

################################### SAMU ############################################
# FUNCTION FOR DETECTING IF THUMB IS UP 👍
def is_thumb_up(handLms, img_shape):
    h, w, c = img_shape
    
    thumb_tip = handLms.landmark[4]
    thumb_mcp = handLms.landmark[2]
    index_finger_tip = handLms.landmark[8]
    middle_finger_tip = handLms.landmark[12]
    ring_finger_tip = handLms.landmark[16]
    pinky_tip = handLms.landmark[20]

    thumb_tip_y = int(thumb_tip.y * h)
    thumb_mcp_y = int(thumb_mcp.y * h)
    index_finger_tip_y = int(index_finger_tip.y * h)
    middle_finger_tip_y = int(middle_finger_tip.y * h)
    ring_finger_tip_y = int(ring_finger_tip.y * h)
    pinky_tip_y = int(pinky_tip.y * h)

    # Ensure all fingers (except thumb) are curled
    fingers = {
        'index': [handLms.landmark[i] for i in [5, 6, 7, 8]],
        'middle': [handLms.landmark[i] for i in [9, 10, 11, 12]],
        'ring': [handLms.landmark[i] for i in [13, 14, 15, 16]],
        'pinky': [handLms.landmark[i] for i in [17, 18, 19, 20]],
    }

    all_fingers_curled = True
    for finger_name, landmarks in fingers.items():
        mcp = (int(landmarks[0].x * w), int(landmarks[0].y * h))
        pip = (int(landmarks[1].x * w), int(landmarks[1].y * h))
        dip = (int(landmarks[2].x * w), int(landmarks[2].y * h))
        tip = (int(landmarks[3].x * w), int(landmarks[3].y * h))
        
        if not is_finger_curled(mcp, pip, dip, tip):
            all_fingers_curled = False
            break

    if all_fingers_curled and (thumb_tip_y < thumb_mcp_y) and (index_finger_tip_y > thumb_mcp_y) and (middle_finger_tip_y > thumb_mcp_y) and (ring_finger_tip_y > thumb_mcp_y) and (pinky_tip_y > thumb_mcp_y):
        return True
    return False

################################### SAMU ############################################
# FUNCTION FOR DETECTING IF THUMB IS DOWN 👎
def is_thumb_down(handLms, img_shape):
    h, w, c = img_shape
    
    thumb_tip = handLms.landmark[4]
    thumb_mcp = handLms.landmark[2]
    index_finger_tip = handLms.landmark[8]
    middle_finger_tip = handLms.landmark[12]
    ring_finger_tip = handLms.landmark[16]
    pinky_tip = handLms.landmark[20]

    thumb_tip_y = int(thumb_tip.y * h)
    thumb_mcp_y = int(thumb_mcp.y * h)
    index_finger_tip_y = int(index_finger_tip.y * h)
    middle_finger_tip_y = int(middle_finger_tip.y * h)
    ring_finger_tip_y = int(ring_finger_tip.y * h)
    pinky_tip_y = int(pinky_tip.y * h)

    # Ensure all fingers (except thumb) are curled
    fingers = {
        'index': [handLms.landmark[i] for i in [5, 6, 7, 8]],
        'middle': [handLms.landmark[i] for i in [9, 10, 11, 12]],
        'ring': [handLms.landmark[i] for i in [13, 14, 15, 16]],
        'pinky': [handLms.landmark[i] for i in [17, 18, 19, 20]],
    }

    all_fingers_curled = True
    for finger_name, landmarks in fingers.items():
        mcp = (int(landmarks[0].x * w), int(landmarks[0].y * h))
        pip = (int(landmarks[1].x * w), int(landmarks[1].y * h))
        dip = (int(landmarks[2].x * w), int(landmarks[2].y * h))
        tip = (int(landmarks[3].x * w), int(landmarks[3].y * h))
        
        if not is_finger_curled(mcp, pip, dip, tip):
            all_fingers_curled = False
            break

    if all_fingers_curled and (thumb_tip_y > thumb_mcp_y) and (index_finger_tip_y < thumb_mcp_y) and (middle_finger_tip_y < thumb_mcp_y) and (ring_finger_tip_y < thumb_mcp_y) and (pinky_tip_y < thumb_mcp_y):
        return True
    return False

################################### SAMU ############################################
# FUNCTION FOR DETECTING RAISED FIST ✊
def is_raised_fist(handLms, img_shape):
    h, w, c = img_shape
    
    thumb_tip = handLms.landmark[4]
    thumb_pip = handLms.landmark[3]
    index_finger_tip = handLms.landmark[8]
    index_finger_mcp = handLms.landmark[5]
    middle_finger_tip = handLms.landmark[12]
    middle_finger_mcp = handLms.landmark[9]
    ring_finger_tip = handLms.landmark[16]
    ring_finger_mcp = handLms.landmark[13]
    pinky_tip = handLms.landmark[20]
    pinky_mcp = handLms.landmark[17]

    thumb_tip_y = int(thumb_tip.y * h)
    thumb_pip_y = int(thumb_pip.y * h)
    index_finger_tip_y = int(index_finger_tip.y * h)
    index_finger_mcp_y = int(index_finger_mcp.y * h)
    middle_finger_tip_y = int(middle_finger_tip.y * h)
    middle_finger_mcp_y = int(middle_finger_mcp.y * h)
    ring_finger_tip_y = int(ring_finger_tip.y * h)
    ring_finger_mcp_y = int(ring_finger_mcp.y * h)
    pinky_tip_y = int(pinky_tip.y * h)
    pinky_mcp_y = int(pinky_mcp.y * h)

    # Get the coordinates of the finger joints in pixel space
    fingers = {
        'thumb': [handLms.landmark[i] for i in [1, 2, 3, 4]],
        'index': [handLms.landmark[i] for i in [5, 6, 7, 8]],
        'middle': [handLms.landmark[i] for i in [9, 10, 11, 12]],
        'ring': [handLms.landmark[i] for i in [13, 14, 15, 16]],
        'pinky': [handLms.landmark[i] for i in [17, 18, 19, 20]],
    }

    all_fingers_curled = True
    for finger_name, landmarks in fingers.items():
        mcp = (int(landmarks[0].x * w), int(landmarks[0].y * h))
        pip = (int(landmarks[1].x * w), int(landmarks[1].y * h))
        dip = (int(landmarks[2].x * w), int(landmarks[2].y * h))
        tip = (int(landmarks[3].x * w), int(landmarks[3].y * h))
        
        if not is_finger_curled(mcp, pip, dip, tip):
            all_fingers_curled = False
            break
        
    if all_fingers_curled and (index_finger_mcp_y < index_finger_tip_y) and (middle_finger_mcp_y < middle_finger_tip_y) and (ring_finger_mcp_y < ring_finger_tip_y) and (pinky_mcp_y < pinky_tip_y) :
        return True
    return False

################################### SAMU ############################################
# FUNCTION FOR DETECTING IF RAISE HAND 🤚
def is_raised_hand(handLms, img_shape):
    h, w, c = img_shape
    
    thumb_tip = handLms.landmark[4]
    thumb_pip = handLms.landmark[3]
    index_finger_tip = handLms.landmark[8]
    index_finger_dip = handLms.landmark[7]
    middle_finger_tip = handLms.landmark[12]
    middle_finger_dip = handLms.landmark[11]
    ring_finger_tip = handLms.landmark[16]
    ring_finger_dip = handLms.landmark[15]
    pinky_tip = handLms.landmark[20]
    pinky_dip = handLms.landmark[19]

    thumb_tip_y = int(thumb_tip.y * h)
    thumb_pip_y = int(thumb_pip.y * h)
    index_finger_tip_y = int(index_finger_tip.y * h)
    index_finger_dip_y = int(index_finger_dip.y * h)
    middle_finger_tip_y = int(middle_finger_tip.y * h)
    middle_finger_dip_y = int(middle_finger_dip.y * h)
    ring_finger_tip_y = int(ring_finger_tip.y * h)
    ring_finger_dip_y = int(ring_finger_dip.y * h)
    pinky_tip_y = int(pinky_tip.y * h)
    pinky_dip_y = int(pinky_dip.y * h)

    # Ensure all fingers (except thumb) are curled
    fingers = {
        'thumb': [handLms.landmark[i] for i in [1, 2, 3, 4]],
        'index': [handLms.landmark[i] for i in [5, 6, 7, 8]],
        'middle': [handLms.landmark[i] for i in [9, 10, 11, 12]],
        'ring': [handLms.landmark[i] for i in [13, 14, 15, 16]],
        'pinky': [handLms.landmark[i] for i in [17, 18, 19, 20]],
    }

    all_fingers_curled = False
    for finger_name, landmarks in fingers.items():
        mcp = (int(landmarks[0].x * w), int(landmarks[0].y * h))
        pip = (int(landmarks[1].x * w), int(landmarks[1].y * h))
        dip = (int(landmarks[2].x * w), int(landmarks[2].y * h))
        tip = (int(landmarks[3].x * w), int(landmarks[3].y * h))
        
        if not is_finger_curled(mcp, pip, dip, tip):
            all_fingers_curled = True
            break

    if all_fingers_curled and (thumb_tip_y < thumb_pip_y) and (index_finger_tip_y < index_finger_dip_y) and (middle_finger_tip_y < middle_finger_dip_y) and (ring_finger_tip_y < ring_finger_dip_y) and (pinky_tip_y < pinky_dip_y):
        return True
    return False

################################### SAMU ############################################
# FUNCTION FOR DETECTING LOVE YOU SIGNAL 🤟
def is_love_you(handLms, img_shape):
    h, w, c = img_shape
    
    thumb_tip = handLms.landmark[4]
    thumb_pip = handLms.landmark[3]
    index_finger_tip = handLms.landmark[8]
    index_finger_dip = handLms.landmark[7]
    middle_finger_tip = handLms.landmark[12]
    middle_finger_mcp = handLms.landmark[9]
    ring_finger_tip = handLms.landmark[16]
    ring_finger_mcp = handLms.landmark[13]
    pinky_tip = handLms.landmark[20]
    pinky_dip = handLms.landmark[19]

    thumb_tip_y = int(thumb_tip.y * h)
    thumb_pip_y = int(thumb_pip.y * h)
    index_finger_tip_y = int(index_finger_tip.y * h)
    index_finger_dip_y = int(index_finger_dip.y * h)
    middle_finger_tip_y = int(middle_finger_tip.y * h)
    middle_finger_mcp_y = int(middle_finger_mcp.y * h)
    ring_finger_tip_y = int(ring_finger_tip.y * h)
    ring_finger_mcp_y = int(ring_finger_mcp.y * h)
    pinky_tip_y = int(pinky_tip.y * h)
    pinky_dip_y = int(pinky_dip.y * h)

    # Get the coordinates of the finger joints in pixel space
    fingers = {
        'middle': [handLms.landmark[i] for i in [9, 10, 11, 12]],
        'ring': [handLms.landmark[i] for i in [13, 14, 15, 16]],
    }

    all_fingers_curled = True
    for finger_name, landmarks in fingers.items():
        mcp = (int(landmarks[0].x * w), int(landmarks[0].y * h))
        pip = (int(landmarks[1].x * w), int(landmarks[1].y * h))
        dip = (int(landmarks[2].x * w), int(landmarks[2].y * h))
        tip = (int(landmarks[3].x * w), int(landmarks[3].y * h))
        
        if not is_finger_curled(mcp, pip, dip, tip):
            all_fingers_curled = False
            break
        
    if all_fingers_curled and (index_finger_dip_y > index_finger_tip_y) and (pinky_dip_y > pinky_tip_y) :
        return True
    return False

################################### SAMU ############################################
# FUNCTION FOR DETECTING IF OKAY HAND 👌
def is_okay_hand(handLms, img_shape):
    h, w, c = img_shape
    
    thumb_tip = handLms.landmark[4]
    thumb_pip = handLms.landmark[3]
    index_finger_tip = handLms.landmark[8]
    index_finger_dip = handLms.landmark[7]
    middle_finger_tip = handLms.landmark[12]
    middle_finger_dip = handLms.landmark[11]
    ring_finger_tip = handLms.landmark[16]
    ring_finger_dip = handLms.landmark[15]
    pinky_tip = handLms.landmark[20]
    pinky_dip = handLms.landmark[19]

    thumb_tip_y = int(thumb_tip.y * h)
    thumb_pip_y = int(thumb_pip.y * h)
    index_finger_tip_y = int(index_finger_tip.y * h)
    index_finger_dip_y = int(index_finger_dip.y * h)
    middle_finger_tip_y = int(middle_finger_tip.y * h)
    middle_finger_dip_y = int(middle_finger_dip.y * h)
    ring_finger_tip_y = int(ring_finger_tip.y * h)
    ring_finger_dip_y = int(ring_finger_dip.y * h)
    pinky_tip_y = int(pinky_tip.y * h)
    pinky_dip_y = int(pinky_dip.y * h)

    # Ensure all fingers (except thumb) are curled
    fingers = {
        'thumb': [handLms.landmark[i] for i in [1, 2, 3, 4]],
        'index': [handLms.landmark[i] for i in [5, 6, 7, 8]],
        #'middle': [handLms.landmark[i] for i in [9, 10, 11, 12]],
        #'ring': [handLms.landmark[i] for i in [13, 14, 15, 16]],
        #'pinky': [handLms.landmark[i] for i in [17, 18, 19, 20]],
    }

    all_fingers_curled = True
    for finger_name, landmarks in fingers.items():
        mcp = (int(landmarks[0].x * w), int(landmarks[0].y * h))
        pip = (int(landmarks[1].x * w), int(landmarks[1].y * h))
        dip = (int(landmarks[2].x * w), int(landmarks[2].y * h))
        tip = (int(landmarks[3].x * w), int(landmarks[3].y * h))
        
        if not is_finger_curled(mcp, pip, dip, tip):
            all_fingers_curled = False
            break

    if all_fingers_curled and (middle_finger_tip_y < middle_finger_dip_y) and (ring_finger_tip_y < ring_finger_dip_y) and (pinky_tip_y < pinky_dip_y):
        return True
    return False

################################### SAMU ############################################
#METHODS TO DECREASE SPEED
#def decrease_speed():
#    global speed
#    if speed > 0:
#        speed -= 50
#        speed = max(speed,0)
#    return speed

################################### SAMU ############################################
#METHODS TO INCREASE SPEED
#def increase_speed():
#    global speed
#    if speed < 255:
#        speed += 50
#        speed = min(speed,255)
#    return speed

################################### SAMU ############################################
# FUNCTION TO DETECT HAND GESTURES
def detect_hand_gesture(handLms, img_shape):
    if is_thumb_up(handLms, img_shape):
        return "thumb-up", 'Speed Up'
    elif is_thumb_down(handLms, img_shape):
        return "thumb-down", 'Speed Down'
    elif is_love_you(handLms, img_shape):
        return "love-you", 'Turn On'
    elif is_raised_hand(handLms, img_shape):
        return "raised-hand", 'Turn Off'
    elif is_okay_hand(handLms, img_shape):
        return "okay-hand", 'Specific Speed'
    elif is_raised_fist(handLms, img_shape):
        return "none", 'Changing Signal'
    return None, None

def send_data_to_arduino(current_gesture, speed):
    if current_gesture == "thumb-up":
        if speed < 255:  # Check if speed is less than the max value
            speed += 40
            arduino.write(b'3')  # Gesture '3' for increasing speed
            arduino.write(str(speed).encode())  # Send speed as a string
    elif current_gesture == "thumb-down":
        if speed > 135:  # Check if speed is greater than 135 to avoid speed down against minimum speed
            speed -= 40
            arduino.write(b'2')  # Gesture '2' for decreasing speed
            arduino.write(str(speed).encode())  # Send speed as a string
    elif current_gesture == "love-you":
        arduino.write(b'1')  # Gesture '1' for medium speed (Turn On)
        arduino.write(str(onSpeed).encode())  # Send speed as a string
    elif current_gesture == "raised-hand":
        arduino.write(b'0')  # Gesture '0' to stop the motor (Turn Off)
        offSpeed = 0
        speed = 135
        arduino.write(str(offSpeed).encode())  # Send speed as a string
    elif current_gesture == "okay-hand":
        arduino.write(b'4')  # Send gesture '4' for medium speed (or as you define)
        medSpeed = 215
        speed = 215
        arduino.write(str(medSpeed).encode())  # Send speed as a string
    #elif is_raised_fist(handLms, img.shape):
        #current_gesture = "none"

    # Optionally wait for a response from Arduino
    response = arduino.readline().decode('utf-8').strip()
    if response:
        print(f"Arduino response: {response}")
    else:
        print("No response from Arduino")

    return speed  # Return updated speed

################################### SAMU ############################################
# MAIN CODE
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            img_shape = img.shape

            # Detect the current gesture and corresponding message
            current_gesture, display_message = detect_hand_gesture(handLms, img_shape)

            if current_gesture and current_gesture != last_gesture:
                # Display the gesture-related message on the screen
                cv2.putText(img, display_message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 223, 0), 2)
                print(current_gesture)

                # Send data to Arduino and update speed if necessary
                speed = send_data_to_arduino(current_gesture, speed)

                # Update the last gesture
                last_gesture = current_gesture

            # Draw landmarks
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()