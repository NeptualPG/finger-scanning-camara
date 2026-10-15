import cv2
import mediapipe as mp


def initialize():
    """Initialize MediaPipe hands and camera."""
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
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

        count = 0

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                points = hand.landmark

                # Four fingers: compare fingertip with middle joint
                for tip, joint in [(8, 6), (12, 10),
                                   (16, 14), (20, 18)]:
                    if points[tip].y < points[joint].y:
                        count += 1

                # Thumb: compare horizontal position
                if points[4].x < points[3].x:
                    count += 1

                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(
            frame, f"Fingers: {count}", (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5,
            (0, 255, 0), 3
        )

        cv2.imshow("Finger Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()