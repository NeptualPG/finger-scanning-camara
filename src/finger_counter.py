import cv2
import mediapipe as mp


def apply_dark_harmony(frame):
    """Give the camera feed a dark blue-violet cinematic appearance."""
    dark_tint = frame.copy()
    dark_tint[:, :, 0] = cv2.add(dark_tint[:, :, 0], 18)
    dark_tint[:, :, 1] = cv2.multiply(dark_tint[:, :, 1], 0.78)
    dark_tint[:, :, 2] = cv2.multiply(dark_tint[:, :, 2], 0.68)

    filtered = cv2.addWeighted(frame, 0.35, dark_tint, 0.65, -8)
    overlay = filtered.copy()
    height, width = filtered.shape[:2]
    cv2.rectangle(overlay, (0, 0), (width, height), (18, 12, 38), -1)
    return cv2.addWeighted(filtered, 0.82, overlay, 0.18, 0)


def draw_styled_text(frame, text, position, color, scale=1.0, thickness=2):
    """Draw readable text with a soft shadow and a clean dark outline."""
    font = cv2.FONT_HERSHEY_DUPLEX
    x, y = position
    cv2.putText(
        frame, text, (x + 2, y + 2), font, scale, (5, 8, 18), thickness + 3
    )
    cv2.putText(
        frame, text, (x, y), font, scale, (22, 28, 48), thickness + 1
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


def draw_hud_panel(frame, total_hands, total_fingers):
    """Draw the top-left status panel over the filtered camera feed."""
    panel = frame.copy()
    cv2.rectangle(panel, (16, 16), (300, 126), (13, 18, 35), -1)
    cv2.rectangle(panel, (16, 16), (300, 126), (92, 220, 255), 2)
    cv2.line(panel, (32, 64), (284, 64), (65, 91, 135), 1)
    cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)

    draw_styled_text(frame, "HAND SCANNER", (32, 49), (170, 238, 255), 0.62, 1)
    draw_styled_text(frame, f"HANDS  {total_hands}", (32, 91), (255, 235, 130), 0.72, 2)
    draw_styled_text(
        frame, f"FINGERS  {total_fingers}", (32, 117), (130, 255, 190), 0.62, 2
    )


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
        frame = apply_dark_harmony(frame)
        height, width = frame.shape[:2]

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

                # Draw landmarks and individual count.
                landmark_style = mp_draw.DrawingSpec(
                    color=(255, 190, 90), thickness=2, circle_radius=3
                )
                connection_style = mp_draw.DrawingSpec(
                    color=(100, 235, 255), thickness=2, circle_radius=2
                )
                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_style,
                    connection_style,
                )

                wrist = points[0]
                x = int(wrist.x * width)
                y = int(wrist.y * height)

                draw_styled_text(
                    frame,
                    str(finger_count),
                    (x, max(34, y - 20)),
                    (130, 255, 190),
                    1.15,
                    2,
                )

        draw_hud_panel(frame, total_hands, total_fingers)

        draw_styled_text(
            frame, "Q  EXIT", (width - 115, height - 24), (190, 205, 230), 0.58, 1
        )

        cv2.imshow("Multiple Hand Finger Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()