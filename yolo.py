from ultralytics import YOLO
import cv2

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Initialize camera (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # YOLO inference on the frame
        results = model(frame)

        # Visualize results on the frame
        annotated_frame = results[0].plot()

        # Display the resulting frame
        cv2.imshow('YOLO Camera Detection', annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

