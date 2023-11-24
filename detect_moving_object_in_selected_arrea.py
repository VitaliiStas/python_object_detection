import cv2
import numpy as np

# Set the screen width and height
screen_width = 600  # Adjust as needed
screen_height = 400  # Adjust as needed

# Use the default webcam as the video source
cap = cv2.VideoCapture(0)

# create a background object
backgroundObject = cv2.createBackgroundSubtractorMOG2(history=2)
kernel = np.ones((3, 3), np.uint8)
kernel2 = None

# Variable to keep track of the last detected location
last_location = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = backgroundObject.apply(gray_frame)
    _, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel2, iterations=6)

    # detect the contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frameCopy = frame.copy()

    # Flag to check if the object is currently being tracked
    object_tracked = False

    # loop inside the contour and search for bigger ones
    for cnt in contours:
        if cv2.contourArea(cnt) > 20000:
            # get the area coordinates
            x, y, width, height = cv2.boundingRect(cnt)

            # draw a rectangle around the area
            cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 0, 255), 2)

            # write a text near the object
            cv2.putText(frameCopy, "Car detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                        cv2.LINE_AA)

            # Update the last detected location
            last_location = (x, y, width, height)
            object_tracked = True

    # Draw a blue frame at the last detected location if the object is not currently being tracked
    if not object_tracked and last_location is not None:
        x, y, width, height = last_location
        cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (255, 0, 0), 2)

        # Print coordinates to the console
        print(f"Last Tracked Position: X={x}, Y={y}, Width={width}, Height={height}")

    cv2.imshow("Tracking Rectangle", cv2.resize(frameCopy, (screen_width, screen_height)))  # Display in full-screen mode

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
