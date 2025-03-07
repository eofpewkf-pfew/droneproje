import tensorflow as tf
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '0' for all, '1' for less, '2' for warnings, '3' for errors

import cv2
import numpy as np

# Load the pre-trained object detection model (enter the path to the saved_model directory)
model_path = "/Users/s1754510/Downloads/centernet_hg104_512x512_coco17_tpu-8/SavedModel"  # Change this to your model path
model = tf.saved_model.load(model_path)

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (TensorFlow models expect RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(rgb_frame)
    input_tensor = input_tensor[tf.newaxis,...]

    # Perform object detection
    detections = model(input_tensor)

    # Get detection data
    boxes = detections['detection_boxes'][0].numpy()
    class_ids = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()

    # Loop over all the detected objects
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold (you can adjust)
            box = boxes[i]
            ymin, xmin, ymax, xmax = box

            # Convert from normalized coordinates to pixel values
            h, w, _ = frame.shape
            (startX, startY, endX, endY) = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))

            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Label (optional)
            label = f"Score: {scores[i]:.2f}"
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the detected objects
    cv2.imshow('Object Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV window
cap.release()
cv2.destroyAllWindows()
