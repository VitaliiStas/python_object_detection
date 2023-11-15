import cv2
import time

# Global variables to store the coordinates of the selected rectangle
rect_start = None
rect_end = None
drawing = False

# Global variables for object tracking
tracker = None
tracked_object_info = None
tracking_initialized = False  # New variable to track if the tracker is initialized

# Variables for frame rate calculation
prev_frame_time = 0

def calculate_center(frame):
    if len(frame.shape) == 3:  # Color image
        height, width, _ = frame.shape
    else:  # Grayscale image
        height, width = frame.shape

    return width // 2, height // 2

def calculate_cross_lines_coordinates(frame, center, line_length_cm=0.5, pixels_per_cm=100):
    # Convert line length from cm to pixels
    line_length = int(line_length_cm * pixels_per_cm)

    # Calculate coordinates for the crossed lines at the center of the frame
    horizontal_line_start = (center[0] - line_length, center[1])
    horizontal_line_end = (center[0] + line_length, center[1])

    vertical_line_start = (center[0], center[1] - line_length)
    vertical_line_end = (center[0], center[1] + line_length)

    return horizontal_line_start, horizontal_line_end, vertical_line_start, vertical_line_end

def draw_cross_lines(frame, center, rect, line_length_cm=0.5, pixels_per_cm=100, color=(0, 255, 0), thickness=1):
    coordinates = calculate_cross_lines_coordinates(frame, center, line_length_cm, pixels_per_cm)

    # Draw crossed lines at the center of the frame
    cv2.line(frame, coordinates[0], coordinates[1], color, thickness)
    cv2.line(frame, coordinates[2], coordinates[3], color, thickness)

    # Draw the custom-selected rectangle during the selection phase
    if rect[0] and rect[1] and not tracker:
        # Draw the blue rectangle
        cv2.rectangle(frame, rect[0], rect[1], (255, 0, 0), thickness)

        # Draw crossed lines within the selected rectangle
        draw_cross_lines_within_rectangle(frame, rect, line_length_cm, pixels_per_cm, color, thickness)

def draw_cross_lines_within_rectangle(frame, rect, line_length_cm, pixels_per_cm, color, thickness):
    center = calculate_center(frame)
    coordinates = calculate_cross_lines_coordinates(frame, center, line_length_cm, pixels_per_cm)

    # Draw crossed lines within the selected rectangle
    cv2.line(frame, coordinates[0], coordinates[1], color, thickness)
    cv2.line(frame, coordinates[2], coordinates[3], color, thickness)

# Mouse event callback
def on_mouse(event, x, y, flags, param):
    global rect_start, rect_end, drawing, tracker, tracked_object_info, tracking_initialized

    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
        drawing = True
        tracking_initialized = False  # Reset the tracker initialization flag

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rect_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        # Initialize the tracker when the selection is complete
        if rect_start and rect_end and not tracking_initialized:
            if rect_start != rect_end:  # Ensure the rectangle has non-zero size
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (rect_start[0], rect_start[1], rect_end[0] - rect_start[0], rect_end[1] - rect_start[1]))

                # Save information about the tracked object, including FPS as the first recording
                tracked_object_info = {
                    'object': 'Tracking object',
                    'first_detected_coordinates': (rect_start[0], rect_start[1]),
                    'last_detected_coordinates': (rect_end[0], rect_end[1]),
                    'fps': 0  # Initialize FPS to 0
                }

                tracking_initialized = True  # Set the tracker initialization flag
            else:
                print("Invalid rectangle size. Please select a valid rectangle.")
        elif not rect_start or not rect_end:
            print("Invalid rectangle coordinates. Please select a valid rectangle.")
        else:
            print("Tracker already initialized.")

# Open a video capture object (you can change the argument to the video file path or camera index)
cap = cv2.VideoCapture(0)

# Assuming pixels_per_cm is 100, adjust it based on your requirement
pixels_per_cm = 100

# Create a window named "Select Rectangle" and set the mouse callback
cv2.namedWindow("Select Rectangle")
cv2.setMouseCallback("Select Rectangle", on_mouse)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame back to color for displaying
    display_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    center = calculate_center(frame)

    # Draw crossed lines at the center of the frame
    draw_cross_lines(display_frame, center, (rect_start, rect_end), line_length_cm=0.5, pixels_per_cm=pixels_per_cm)

    new_frame_time = time.time()

    # Calculate and update FPS only if the time difference is not zero
    if new_frame_time != prev_frame_time:
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        # If a tracker is not initialized, calculate and display FPS
        if not tracking_initialized:
            # Display FPS information on the frame
            cv2.putText(display_frame, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # If a tracker is initialized, update and draw the tracked object
        if tracking_initialized:
            # Convert grayscale frame back to color before passing to the tracker
            color_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            success, bbox = tracker.update(color_frame)
            if success:
                bbox = tuple(map(int, bbox))
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 1)

                # Update information about the tracked object
                tracked_object_info['last_detected_coordinates'] = (bbox[0] + bbox[2], bbox[1] + bbox[3])

                # Calculate and update FPS
                new_frame_time = time.time()
                if new_frame_time != prev_frame_time:  # Check for zero division
                    tracked_object_info['fps'] = int(1 / (new_frame_time - prev_frame_time))
                    prev_frame_time = new_frame_time

                    # Display information about the tracked object and FPS in the console
                    print(tracked_object_info)
                    # Display FPS information on the frame
                    cv2.putText(display_frame, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # Display the text 'last_detected_coordinates' below 'FPS'
                    cv2.putText(display_frame, f"last_detected_coordinates: {tracked_object_info['last_detected_coordinates']}",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Display "Object Detected" text on the frame
                    cv2.putText(display_frame, "Object Detected", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                1)

    # Display the frame with the crossed lines
    cv2.imshow("Select Rectangle", display_frame)

    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(30) == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
