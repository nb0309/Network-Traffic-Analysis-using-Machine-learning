import cv2
import numpy as np

def identify_green(frame):
    """Detects green objects in the frame and sends a signal to a solenoid valve (Raspberry Pi specific).

    Args:
        frame (numpy.ndarray): The frame captured from the video stream.

    Returns:
        None
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    green_detected = False
    for cnt in contours:
        # Calculate area of the contour
        area = cv2.contourArea(cnt)
        # Adjust minimum area threshold as needed
        if area > 1000:  # Adjust threshold based on object size and camera distance
            green_detected = True
            cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 2)
            break  # Exit loop after finding one green object

    # Send signal to solenoid valve (Raspberry Pi specific)
    

    cv2.imshow("Green Detection", frame)




cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        identify_green(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Frame not captured")
        break

cap.release()
cv2.destroyAllWindows()
