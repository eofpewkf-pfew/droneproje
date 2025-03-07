import tensorflow as tf
import cv2
import numpy as np
from codrone_edu.drone import *
import time
import math

# Initialize the drone
drone = Drone()
drone.pair()
print("Paired!")

# Load the pre-trained object detection model
model_path = "/path/to/your/saved_model"
model = tf.saved_model.load(model_path)

# Start the drone
drone.takeoff()
print("In the air!")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Movement Speed (Adjust based on your preferences)
MOVEMENT_SPEED = 20
ROTATION_SPEED = 10
CIRCLE_RADIUS = 50  # Define the radius of the circular motion

def circle_around_room():
    """ Make the drone move in a circular path """
    while True:
        drone.forward(MOVEMENT_SPEED)  # Move forward
        drone.turn_right(ROTATION_SPEED)  # Rotate right slightly
        time.sleep(0.1)  # Adjust this to control the speed of the circle
        # This loop will make the drone move in a circle as it keeps moving forward and turning right

def move_towards_object(center_x, center_y, frame_center_x, frame_center_y, w, h):
    """ Move drone towards the detected object based on its position """
    if center_x < frame_center_x - 50:  # Object is left of the frame center
        drone.turn_left(ROTATION_SPEED)
    elif center_x > frame_center_x + 50:  # Object is right of the frame center
        drone.turn_right(ROTATION_SPEED)

    if center_y < frame_center_y - 50:  # Object is above the center
        drone.ascend(MOVEMENT_SPEED)  # Using ascend to move up
    elif center_y > frame_center_y + 50:  # Object is below the center
        drone.descend(MOVEMENT_SPEED)  # Using descend to move down

def zigzag_movement():
    """ Make the drone fly in a zigzag pattern when no object is detected """
    drone.forward(MOVEMENT_SPEED)
    time.sleep(1)
    drone.turn_right(ROTATION_SPEED)
    time.sleep(0.5)
    drone.forward(MOVEMENT_SPEED)
    time.sleep(1)
    drone.turn_left(ROTATION_SPEED)
    time.sleep(0.5)

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
                break  # Exit loop after first detection (you can modify for multiple objects)

    if not object_detected:
        # If no object is detected, make the drone perform a zigzag movement
        print("No object detected. Performing zigzag movement.")
        zigzag_movement()

    # Display the frame with detected objects
    cv2.imshow('Object Detection and Drone Control', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Start the circular movement
print("Starting circular movement.")
circle_around_room()

# Land the drone once the loop ends
drone.land()
print("Landing")
drone.close()
print("Program complete")

# Release the webcam and close OpenCV window
cap.release()
cv2.destroyAllWindows()
