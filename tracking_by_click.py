import cv2
import numpy as np
import time

# Global variables to store the coordinates of the selected rectangle
rect_start = None
rect_end = None
drawing = False

# Global variables for object tracking
tracker = None
tracked_object_info = None
tracking_initialized = False  # New variable to track if the tracker is initialized
pixels_per_cm=100
# Variables for frame rate calculation
prev_frame_time = 0
frame = None  # Global variable to store the current frame

# Mouse event callback
def on_mouse(event, x, y, flags, param):
    global rect_start, rect_end, drawing, tracker, tracked_object_info, tracking_initialized

    # Extract the pixels_per_cm parameter from param
    pixels_per_cm = param.get('pixels_per_cm', 100)

    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
        drawing = True
        tracking_initialized = False

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        if rect_start and not tracking_initialized:
            initialize_tracker((x, y), pixels_per_cm)  # Pass the center coordinates as a parameter
        elif not rect_start:
            print("Invalid starting point. Please click within the frame.")
        else:
            print("Tracker already initialized.")

# Function to initialize the tracker for object tracking
def initialize_tracker(center, size):
    global tracker, tracked_object_info, tracking_initialized
    rect_half_size = size // 2
    rect_start = (center[0] - rect_half_size, center[1] - rect_half_size)
    rect_end = (rect_start[0] + size, rect_start[1] + size)

    if rect_start != rect_end:
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, (rect_start[0], rect_start[1], rect_end[0] - rect_start[0], rect_end[1] - rect_start[1]))

        tracked_object_info = {
            'object': 'Tracking object',
            'first_detected_coordinates': (rect_start[0], rect_start[1]),
            'last_detected_coordinates': (rect_end[0], rect_end[1]),
            'fps': 0
        }
        tracking_initialized = True
    else:
        print("Invalid rectangle size. Please select a valid rectangle.")

# Function to draw crossed lines at the center of the frame
def draw_cross_lines(frame, center, rect, line_length_cm=1, pixels_per_cm=pixels_per_cm, color=(0, 255, 0), thickness=1):
    horizontal_line_start, horizontal_line_end = calculate_crossed_lines(center, line_length_cm, pixels_per_cm, horizontal=True)
    vertical_line_start, vertical_line_end = calculate_crossed_lines(center, line_length_cm, pixels_per_cm, horizontal=False)

    # Draw crossed lines
    draw_line(frame, horizontal_line_start, horizontal_line_end, color, thickness)
    draw_line(frame, vertical_line_start, vertical_line_end, color, thickness)

    # Draw the custom-selected rectangle during the selection phase
    if rect[0] and rect[1] and not tracker:
        # Calculate the size of the rectangle (1 cm)
        rect_size = int(pixels_per_cm)
        # Calculate the top-left point of the rectangle
        rect_start = (rect[0] - rect_size // 2, rect[1] - rect_size // 2)
        # Draw the red rectangle
        draw_rectangle(frame, rect_start, rect_size, (0, 0, 255), thickness)
        # Draw crossed lines within the selected rectangle
        draw_cross_lines_within_rectangle(frame, (rect_start, (rect_start[0] + rect_size, rect_start[1] + rect_size)),
                                          line_length_cm, pixels_per_cm, color, thickness)

# Function to draw crossed lines within a rectangle
def draw_cross_lines_within_rectangle(frame, rect, line_length_cm, pixels_per_cm, color, thickness):
    center = calculate_rectangle_center(rect)

    # Draw crossed lines within the selected rectangle
    horizontal_line_start, horizontal_line_end = calculate_crossed_lines(center, line_length_cm, pixels_per_cm, horizontal=True)
    vertical_line_start, vertical_line_end = calculate_crossed_lines(center, line_length_cm, pixels_per_cm, horizontal=False)

    draw_line(frame, horizontal_line_start, horizontal_line_end, color, thickness)
    draw_line(frame, vertical_line_start, vertical_line_end, color, thickness)

    # Draw the non-rotated rectangle
    draw_rectangle(frame, rect[0], rect[1][0] - rect[0][0], (0, 0, 255), thickness)

# Function to calculate crossed lines given a center point, line length, and pixels per cm
def calculate_crossed_lines(center, line_length_cm, pixels_per_cm, horizontal=True):
    offset = int(line_length_cm * pixels_per_cm / 2)
    if horizontal:
        return (center[0] - offset, center[1]), (center[0] + offset, center[1])
    else:
        return (center[0], center[1] - offset), (center[0], center[1] + offset)

# Function to draw a line on the frame
def draw_line(frame, start, end, color, thickness):
    cv2.line(frame, tuple(map(int, start)), tuple(map(int, end)), color, thickness)

# Function to draw a rectangle on the frame
def draw_rectangle(frame, start, size, color, thickness):
    cv2.rectangle(frame, tuple(map(int, start)), (start[0] + size, start[1] + size), color, thickness)

# Function to calculate the center of a rectangle
def calculate_rectangle_center(rect):
    return (rect[0][0] + (rect[1][0] - rect[0][0]) // 2, rect[0][1] + (rect[1][1] - rect[0][1]) // 2)

# Function to update and draw the tracked object
def update_tracked_object(frame, bbox):
    global tracked_object_info, prev_frame_time

    bbox = tuple(map(int, bbox))
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 1)

    tracked_object_info['last_detected_coordinates'] = (bbox[0] + bbox[2], bbox[1] + bbox[3])

    new_frame_time = time.time()
    if new_frame_time != prev_frame_time:
        tracked_object_info['fps'] = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        # Display information about the tracked object and FPS in the console
        print(tracked_object_info)

        # Display FPS information on the frame
        cv2.putText(frame, f"FPS: {tracked_object_info['fps']}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Display the text 'last_detected_coordinates' below 'FPS'
        cv2.putText(frame, f"last_detected_coordinates: {tracked_object_info['last_detected_coordinates']}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw a circle at the center of the tracking frame
        draw_circle_in_center(frame, bbox, circle_diameter_cm=0.5, pixels_per_cm=pixels_per_cm)

        # Draw "Object Detected" text on the frame
        draw_object_detected_text(frame, bbox)

# Function to draw a circle at the center of the tracking frame
def draw_circle_in_center(frame, bbox, circle_diameter_cm, pixels_per_cm):
    center_x = int(bbox[0] + bbox[2] / 2)
    center_y = int(bbox[1] + bbox[3] / 2)

    # Calculate the radius for a circle with the given diameter
    circle_radius = int(circle_diameter_cm * pixels_per_cm / 2)

    # Draw a circle at the center of the tracking frame
    cv2.circle(frame, (center_x, center_y), circle_radius, (0, 0, 255), 1)

# Function to draw "Object Detected" text on the frame
def draw_object_detected_text(frame, bbox):
    # Draw "Object Detected" text on the frame
    cv2.putText(frame, "Object Detected", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Calculate the center coordinates of the tracking frame
    center_x = int(bbox[0] + bbox[2] / 2)
    center_y = int(bbox[1] + bbox[3] / 2)

    # Display the center coordinates of the tracking frame as x and y
    cv2.putText(frame, f"x: {center_x}, y: {center_y}",
                (bbox[0], bbox[1] + bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Main function
def main():
    global rect_start, rect_end, drawing, tracker, tracked_object_info, tracking_initialized, prev_frame_time, frame

    # Open a video capture object (you can change the argument to the video file path or camera index)
    # cap = cv2.VideoCapture('test2.MP4')
    cap = cv2.VideoCapture(0)

    # Set the frame rate (not supported by all codecs)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Set to 60 frames per second



    # Create a window named "Select Rectangle" and set the mouse callback
    cv2.namedWindow("Select Rectangle")
    cv2.setMouseCallback("Select Rectangle", on_mouse, {'pixels_per_cm': pixels_per_cm})

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert grayscale frame back to color for displaying
        display_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        center = (frame.shape[1] // 2, frame.shape[0] // 2)

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
                    update_tracked_object(display_frame, bbox)

        # Display the frame with the crossed lines
        cv2.imshow("Select Rectangle", display_frame)

        # Break the loop if the 'Esc' key is pressed
        if cv2.waitKey(30) == 27:
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
