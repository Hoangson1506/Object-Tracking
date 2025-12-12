import cv2
import numpy as np
import supervision as sv

def draw_polygon_zone(frame: np.ndarray, window_name: str = "Traffic Violation Detection"):
    """
    Interactive mode to draw a Polygon Zone (Region of Interest, RoI).
    
    The user uses mouse clicks to define vertices.
    Press 'n' to save the polygon (requires ≥ 3 points). 
    Press 'ESC' to cancel.

    Args:
        frame (np.ndarray): The video frame used as the canvas.

    Return:
        drawing_points: A list of points defining the RoI polygon (N, 2). 
                    Returns [] if cancelled.
    """
    
    zone_name = "POLYGON_ROI"
    drawing_points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing_points
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_points.append((x, y))
            print(f"[{zone_name}] Point selected: ({x}, {y})")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n--- [TUTORIAL] {zone_name} ---")
    print("1. Left mouse click to choose POLYGON points (RoI Region)")
    print("2. Press 's' to save the polygon (Needs ≥ 3 points)")
    print("3. Press 'r' to revert back 1 point")
    print("3. Press 'ESC' to cancel")
    print("Warning: At least 3 points are needed for a polygon. If you save with less than 3 points, it will return None.")


    while True:
        display_frame = frame.copy()

        # Draw selected points
        for pt in drawing_points:
            cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)

        # Draw the polygon if enough points are selected (>= 3)
        if len(drawing_points) >= 3:
            pts = np.array(drawing_points, np.int32).reshape((-1, 1, 2))
            # Draw the connected, closed polygon (True)
            cv2.polylines(display_frame, [pts], True, (255, 0, 0), 2)
        
        cv2.putText(display_frame,
                    f"MODE: {zone_name} | Points: {len(drawing_points)}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print(f"ESC pressed → {zone_name} cancelled")
            cv2.destroyWindow(window_name)
            return []
        
        if key == ord('r'):
            if len(drawing_points) > 0:
                removed_point = drawing_points.pop()
                print(f"[{zone_name}] Removed last point: {removed_point}")
            else:
                print(f"[{zone_name}] No points to remove!")

        if key == ord('s'):
            if len(drawing_points) < 3:
                print(f"[{zone_name}] Requires at least 3 points for a polygon!")
                cv2.destroyWindow(window_name)
                return drawing_points
            
            print(f"[{zone_name}] Polygon saved with {len(drawing_points)} points.")
            cv2.destroyWindow(window_name)
            return drawing_points


def draw_line_zone(frame: np.ndarray, window_name: str = "Traffic Violation Detection"):
    """
    Interactive mode to draw a Line Zone (Boundary/Crossing Line).
    
    The user must define exactly 2 points.
    Press 'q' to save the line. Press 'ESC' to cancel.

    Args:
        frame (np.ndarray): The video frame used as the canvas.

    Return:
        drawing_points: A list of 2 points defining the line. Returns [] if cancelled.
    """
    
    zone_name = "LINE_ZONE"
    violation_lines_drawing_points = []
    special_violation_lines_drawing_points = []
    left_exception_lines_drawing_points = []
    right_exception_lines_drawing_points = []
    other_exception_lines_drawing_points = []
    
    def mouse_callback_for_violation(event, x, y, flags, param):
        nonlocal violation_lines_drawing_points
        if event == cv2.EVENT_LBUTTONDOWN:
            violation_lines_drawing_points.append((x, y))
            print(f"[{zone_name}] Point selected: ({x}, {y})")

    def mouse_callback_for_special_violation(event, x, y, flags, param):
        nonlocal special_violation_lines_drawing_points
        if event == cv2.EVENT_LBUTTONDOWN:
            special_violation_lines_drawing_points.append((x, y))
            print(f"[{zone_name}] Point selected: ({x}, {y})")

    def mouse_callback_for_left_exception(event, x, y, flags, param):
        nonlocal left_exception_lines_drawing_points
        if event == cv2.EVENT_LBUTTONDOWN:
            left_exception_lines_drawing_points.append((x, y))
            print(f"[{zone_name}] Point selected: ({x}, {y})")

    def mouse_callback_for_right_exception(event, x, y, flags, param):
        nonlocal right_exception_lines_drawing_points
        if event == cv2.EVENT_LBUTTONDOWN:
            right_exception_lines_drawing_points.append((x, y))
            print(f"[{zone_name}] Point selected: ({x}, {y})")

    def mouse_callback_for_exception(event, x, y, flags, param):
        nonlocal other_exception_lines_drawing_points
        if event == cv2.EVENT_LBUTTONDOWN:
            other_exception_lines_drawing_points.append((x, y))
            print(f"[{zone_name}] Point selected: ({x}, {y})")

    cv2.namedWindow(window_name)

    print(f"\n--- [TUTORIAL] {zone_name} ---")
    print("1. Left mouse click to choose 2 points for each LINE")
    print("2. Press 'r' to revert back 1 point")
    print("3. Press 'ESC' to cancel")
    print("4. First, draw base violation LINES.")
    print("5. Press 's' to save the violation LINES and move to the next step drawing special violation LINES.\n" \
    "These LINES will make any vehicles crossing it a violated one, used for cases like no U-turn whatever the traffic light state is. \n" \
    "If no special violation LINES, just press 's' to move to the nest step.")
    print("6. Next, draw left turn exception LINES. \n" \
    "These LINES will exempt vehicles crossing it from being marked as violated when the corresponding traffic light is NOT red. \n" \
    "If no left turn exception LINES, just press 's' to move to the next step.")
    print("7. Next, draw right turn exception LINES. They work the same way as left turn exception LINES. \n" \
    "If no right turn exception LINES, just press 's' to move to the next step.")
    print("8. Finally, draw other exception LINES (e.g. U-turn). They always exempt vehicles crossing it from being marked as violated whatever the traffic light state is.\n" \
    "If no other exception LINES, just press 's' to finish.")

    # Mapping from step index to corresponding mouse callback, drawing points list and line color
    i_to_line = {0: [mouse_callback_for_violation, violation_lines_drawing_points, (0, 0, 255)],
                    1: [mouse_callback_for_special_violation, special_violation_lines_drawing_points, (0, 165, 255)],
                    2: [mouse_callback_for_left_exception, left_exception_lines_drawing_points, (255, 0, 0)],
                    3: [mouse_callback_for_right_exception, right_exception_lines_drawing_points, (255, 255, 0)],
                    4: [mouse_callback_for_exception, other_exception_lines_drawing_points, (255, 0, 255)]}
    

    for i in range(5):
        cv2.setMouseCallback(window_name, i_to_line[i][0])
        while True:
            if i == 0:
                display_frame = frame.copy()

            # Draw selected points
            for pt in i_to_line[i][1]:
                cv2.circle(display_frame, pt, 5, i_to_line[i][2], -1)

            # Draw the line if 2 points are selected
            if len(i_to_line[i][1]) >= 2:
                for j in range(0, len(i_to_line[i][1])-1, 2):
                    cv2.line(display_frame, i_to_line[i][1][j], i_to_line[i][1][j+1], i_to_line[i][2], 2)

            cv2.putText(display_frame,
                        f"MODE: {zone_name} | Step: {i+1}/5 | Points: {len(i_to_line[i][1])}",
                        (20, 30 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 0), 2)
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27: # ESC
                print(f"ESC pressed → {zone_name} cancelled")
                cv2.destroyWindow(window_name)
                return [], [], [], [], []
            if key == ord('r'):
                if len(i_to_line[i][1]) > 0:
                    removed_point = i_to_line[i][1].pop()
                    print(f"[{zone_name}] Removed last point: {removed_point}")
                else:
                    print(f"[{zone_name}] No points to remove!")
            if key == ord('s'):
                if i == 0 and len(i_to_line[i][1]) < 2:
                    print(f"[{zone_name}] Requires at least 2 points for a line!")
                elif len(i_to_line[i][1]) % 2 != 0:
                    print(f"[{zone_name}] Requires even number of points for line pairs!")
                else:
                    print(f"[{zone_name}] Step {i+1} lines saved.")
                    break
    
    print(f"[{zone_name}] All lines saved.")
    cv2.destroyWindow(window_name)
    return violation_lines_drawing_points, special_violation_lines_drawing_points, left_exception_lines_drawing_points, right_exception_lines_drawing_points, other_exception_lines_drawing_points


def draw_and_write_frame(tracked_objs, frame, sv_detections, box_annotator, label_annotator, video_writer, window_name="Traffic Violation Detection"):
    """Process a single detection result, draws bbox, writes the frame

    Args:
        tracked_objs (KalmanBoxTracker): List of tracked objects 
        frame (ArrayLike): The frame to write on
        sv_detections (sv.Detections): Detections result in the supervision format
        box_annotator (sv.BoxAnnotator)
        label_annotator (sv.LabelAnnotator)
        video_writer (cv2.VideoWriter)
    """
    frame = box_annotator.annotate(
        scene=frame,
        detections=sv_detections
    )

    labels = [f"ID: {obj.id} {'[VIOLATION]' if obj.has_violated else ''}" for obj in tracked_objs]
    frame = label_annotator.annotate(
        scene=frame,
        detections=sv_detections,
        labels=labels
    )

    if video_writer is not None:
        video_writer.write(frame)

    cv2.imshow(window_name, frame)
