import os
import logging
import tensorflow as tf
import cv2
import numpy as np
from codrone_edu.drone import *
import time
import random
import threading

# Suppress TensorFlow and absl logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
logging.getLogger('absl').setLevel(logging.ERROR)  # Suppress absl logging

# Initialize the drone
drone = Drone()
drone.pair()
print("Paired!")

# Load the pre-trained object detection model
model_path = "/Users/s1754510/Downloads/centernet_hg104_512x512_coco17_tpu-8/SavedModel"
model = tf.saved_model.load(model_path)

# Start the drone
drone.takeoff()
print("In the air!")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Movement Speed (Adjust based on your preferences)
MOVEMENT_SPEED = 40  # Faster movement for larger rooms
ROTATION_SPEED = 15
CIRCLE_RADIUS = 100  # Increase the radius of the circular motion for a larger area

# Variables to manage drone's motion
object_detected = False

def random_move():
    """ Make the drone move randomly in different directions over a larger area """
    move_choice = random.choice(['move_forward', 'move_backward', 'turn_left', 'turn_right', 'ascend', 'descend'])
    if move_choice == 'move_forward':
        drone.set_pitch(50)  # Move forward with a positive pitch
        time.sleep(2)  # Increased time for larger movement
        drone.set_pitch(0)  # Stop moving forward
    elif move_choice == 'move_backward':
        drone.set_pitch(-50)  # Move backward with a negative pitch
        time.sleep(2)  # Increased time for larger movement
        drone.set_pitch(0)  # Stop moving backward
    elif move_choice == 'turn_left':
        drone.set_yaw(100)  # Turn left
        time.sleep(2)  # Increased time for a larger turning radius
        drone.set_yaw(0)  # Stop turning
    elif move_choice == 'turn_right':
        drone.set_yaw(-100)  # Turn right
        time.sleep(2)  # Increased time for a larger turning radius
        drone.set_yaw(0)  # Stop turning
    elif move_choice == 'ascend':
        drone.set_throttle(75)  # Ascend with a higher throttle for a bigger room
        time.sleep(2)  # Increased time for larger movement
        drone.set_throttle(0)  # Stop ascending
    elif move_choice == 'descend':
        drone.set_throttle(-75)  # Descend with a higher throttle for a bigger room
        time.sleep(2)  # Increased time for larger movement
        drone.set_throttle(0)  # Stop descending

def creative_move():
    """ Perform creative flight moves like figure-eight or random spins across a larger area """
    move_choice = random.choice(['spin', 'figure_eight', 'ascend', 'descend', 'wobble'])
    if move_choice == 'spin':
        drone.set_yaw(100)  # Spin to the left
        time.sleep(2)  # Increased time for a full spin
        drone.set_yaw(-100)  # Spin to the right
        time.sleep(2)
    elif move_choice == 'figure_eight':
        drone.set_pitch(50)  # Move forward in larger strides
        time.sleep(2)
        drone.set_yaw(-75)  # Turn a bit to the right
        time.sleep(1)
        drone.set_pitch(50)  # Move forward again
        time.sleep(2)
        drone.set_yaw(75)  # Turn a bit to the left
        time.sleep(1)
    elif move_choice == 'ascend':
        drone.set_throttle(75)  # Ascend with a higher throttle
        time.sleep(2)
        drone.set_throttle(0)  # Stop ascending
    elif move_choice == 'descend':
        drone.set_throttle(-75)  # Descend with a higher throttle
        time.sleep(2)
        drone.set_throttle(0)  # Stop descending
    elif move_choice == 'wobble':
        drone.set_yaw(50)  # Wobble left
        time.sleep(1)
        drone.set_yaw(-50)  # Wobble right
        time.sleep(1)

def move_towards_object(center_x, center_y, frame_center_x, frame_center_y, w, h):
    """ Move drone towards the detected object based on its position """
    if center_x < frame_center_x - 100:  # Object is far left of the frame center
        drone.set_yaw(100)  # Turn left (yaw left)
    elif center_x > frame_center_x + 100:  # Object is far right of the frame center
        drone.set_yaw(-100)  # Turn right (yaw right)

    if center_y < frame_center_y - 100:  # Object is far above the center
        drone.set_throttle(75)  # Ascend faster
    elif center_y > frame_center_y + 100:  # Object is far below the center
        drone.set_throttle(-75)  # Descend faster

def avoid_object_and_continue():
    """ If an object is detected, adjust the path but continue moving around the room """
    random_move()  # Randomly change direction to avoid obstacles

def circular_movement():
    """ Keep the drone moving in a larger circular pattern """
    while True:
        drone.set_pitch(50)  # Move forward
        drone.set_yaw(-15)  # Turn right (yaw right)
        time.sleep(0.2)  # Adjust this to control the speed of the circle

# Start circular movement in a separate thread to keep it running in the background
movement_thread = threading.Thread(target=circular_movement)
movement_thread.daemon = True  # This ensures the thread will exit when the program terminates
movement_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (TensorFlow models expect RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(rgb_frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform object detection
    detections = model(input_tensor)

    # Get detection data
    boxes = detections['detection_boxes'][0].numpy()
    class_ids = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()

    object_detected = False  # Flag to check if any object is detected

    # Loop over all detected objects
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold (adjustable)
            box = boxes[i]
            ymin, xmin, ymax, xmax = box

            # Convert from normalized coordinates to pixel values
            h, w, _ = frame.shape
            (startX, startY, endX, endY) = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))

            # Draw rectangle around the detected object
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Optional: Draw label with score
            label = f"Score: {scores[i]:.2f}"
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if the detected object is a person (you can map class_ids to actual labels)
            if class_ids[i] == 1:  # Assuming 1 is the class ID for 'person'
                print("Person detected!")

                # Get the object's center coordinates in the frame
                center_x = (startX + endX) // 2
                center_y = (startY + endY) // 2

                # Frame center
                frame_center_x = w // 2
                frame_center_y = h // 2

                # Move the drone towards the object
                move_towards_object(center_x, center_y, frame_center_x, frame_center_y, w, h)

                object_detected = True
                avoid_object_and_continue()  # Avoid the detected object and keep moving
                break  # Exit loop after first detection (you can modify for multiple objects)

    if not object_detected:
        # If no object is detected, make the drone perform random or creative movements
        print("No object detected. Continuing movement.")
        random_move()

        # Occasionally perform creative flight
        if random.random() < 0.1:  # 10% chance of doing a creative move
            print("Performing creative move.")
            creative_move()

    # Display the frame with detected objects
    cv2.imshow('Object Detection and Drone Control', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Land the drone once the loop ends
drone.land()
print("Landing")
drone.close()
print("Program complete")

# Release the webcam and close OpenCV window
cap.release()
cv2.destroyAllWindows()
