import cv2
import numpy as np

# Set the screen width and height
screen_width = 800  # Adjust as needed
screen_height = 600  # Adjust as needed

# Use the default webcam as the video source
cap = cv2.VideoCapture("test1.MP4")
# cap = cv2.VideoCapture(0)

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Variable to keep track of the last detected location
last_location = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Use Gaussian blur to reduce noise
    fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)

    # Threshold the image to get binary mask
    _, thresh = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_copy = frame.copy()

    # Loop inside the contour and search for moving objects
    for cnt in contours:
        if cv2.contourArea(cnt) > 8000:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw a rectangle around the moving object
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Update the last detected location
            last_location = (x, y, w, h)

    # Draw a blue frame at the last detected location
    if last_location is not None:
        x, y, w, h = last_location
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Print coordinates to the console
        print(f"Last Tracked Position: X={x}, Y={y}, Width={w}, Height={h}")

    cv2.imshow("Motion Detection", cv2.resize(frame_copy, (screen_width, screen_height)))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
