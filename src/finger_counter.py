import cv2
import mediapipe as mp


def initialize():
    """Initialize MediaPipe hands and camera."""
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=10,  # Detect multiple hands
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    camera = cv2.VideoCapture(0)

    return mp_hands, mp_draw, hands, camera


def run():
    mp_hands, mp_draw, hands, camera = initialize()

    while camera.isOpened():
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        total_fingers = 0
        total_hands = 0

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                points = hand.landmark
                finger_count = 0
                total_hands += 1

                # Index, middle, ring, and little fingers
                for tip, joint in [
                    (8, 6), (12, 10), (16, 14), (20, 18)
                ]:
                    if points[tip].y < points[joint].y:
                        finger_count += 1

                # Thumb: account for left/right hand orientation
                handedness = result.multi_handedness[
                    total_hands - 1
                ].classification[0].label

                if handedness == "Right":
                    if points[4].x < points[3].x:
                        finger_count += 1
                else:
                    if points[4].x > points[3].x:
                        finger_count += 1

                total_fingers += finger_count

                # Draw landmarks and individual count
                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                wrist = points[0]
                height, width, _ = frame.shape
                x = int(wrist.x * width)
                y = int(wrist.y * height)

                cv2.putText(
                    frame,
                    str(finger_count),
                    (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )

        # Display totals for every detected hand
        cv2.putText(
            frame,
            f"Hands: {total_hands}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Total fingers: {total_fingers}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Multiple Hand Finger Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()