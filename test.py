import cv2
import time
from ultralytics import YOLO

# Load your YOLO model (provide the correct path to the trained weights)
model = YOLO("weights/best.pt")  # Load model from weights directory

# Initialize video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Desired frames per second (FPS)
desired_fps = 5
frame_interval = 1 / desired_fps

print("Starting live pothole detection. Press 'q' to quit.")

# Main loop for live detection
while cap.isOpened():
    start_time = time.time()  # Record start time for frame processing

    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        break

    # Run YOLO model on the frame
    results = model(frame)

    # Annotate the frame with the detection results
    annotated_frame = results[0].plot()  # Assumes YOLO provides a plot method

    # Display the annotated frame
    cv2.imshow("Live Pothole Detection", annotated_frame)

    # Calculate the elapsed time and wait to maintain desired FPS
    elapsed_time = time.time() - start_time
    time_to_wait = frame_interval - elapsed_time
    if time_to_wait > 0:
        time.sleep(time_to_wait)

    # Break the loop when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting live detection...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Resources released. Detection terminated.")
