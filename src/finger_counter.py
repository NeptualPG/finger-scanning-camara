import cv2

from camera import apply_punk_filter, open_camera, prepare_frame
from hand_tracking import HandTracker
from ui import draw_exit_hint, draw_hand_counts, draw_hud_panel


WINDOW_TITLE = "Multiple Hand Finger Counter"


def run():
    """Run the camera preview and display the live finger counts."""
    camera = open_camera()
    tracker = HandTracker()

    try:
        while camera.isOpened():
            success, frame = camera.read()
            if not success:
                break

            frame = prepare_frame(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = tracker.process(rgb_frame)
            frame = apply_punk_filter(frame)

            observations = tracker.analyze(result)
            tracker.draw_landmarks(frame, observations)
            draw_hand_counts(frame, observations)

            total_hands = len(observations)
            total_fingers = sum(
                observation.finger_count for observation in observations
            )
            draw_hud_panel(frame, total_hands, total_fingers)
            draw_exit_hint(frame)

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
