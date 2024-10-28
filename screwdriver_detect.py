from ultralytics import YOLO
import cv2

# Load the YOLOv11 model
model = YOLO('best_train4.pt')  # Replace with the path to your YOLOv11 .pt file

def detect_objects_from_webcam():
    # Open the webcam
    cap = cv2.VideoCapture(0)  # '0' is the default camera, change if needed

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Perform object detection on the frame
        results = model.predict(frame, conf=0.75)  # Detect objects in the frame

        # Process and draw bounding boxes
        for result in results:
            for box in result.boxes:
                # Extract box coordinates, confidence, and class name
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # xyxy format
                conf = box.conf[0]
                cls = box.cls[0].item()
                label = f"{model.names[int(cls)]}: {conf:.2f}"

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Webcam Object Detection', frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the webcam detection function
detect_objects_from_webcam()