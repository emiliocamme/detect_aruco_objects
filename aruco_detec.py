import cv2

# Load the chosen dictionary. Here we use the 4x4 50 marker set.
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)  # Create an ArUco detector instance


# Start video capture (0 is typically the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    # Draw detected markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        print("Detected IDs:", ids)  # Optionally print the IDs

    # Show the frame with markers
    cv2.imshow("ArUco Marker Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
