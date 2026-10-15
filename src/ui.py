import cv2


def draw_styled_text(frame, text, position, color, scale=1.0, thickness=2):
    """Draw readable text with a soft shadow and a dark outline."""
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
    """Draw the status panel in the top-left corner."""
    panel = frame.copy()
    cv2.rectangle(panel, (16, 16), (300, 126), (13, 18, 35), -1)
    cv2.rectangle(panel, (16, 16), (300, 126), (92, 220, 255), 2)
    cv2.line(panel, (32, 64), (284, 64), (65, 91, 135), 1)
    cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)

    draw_styled_text(frame, "HAND SCANNER", (32, 49), (170, 238, 255), 0.62, 1)
    draw_styled_text(
        frame, f"HANDS  {total_hands}", (32, 91), (255, 235, 130), 0.72, 2
    )
    draw_styled_text(
        frame,
        f"FINGERS  {total_fingers}",
        (32, 117),
        (130, 255, 190),
        0.62,
        2,
    )


def draw_hand_counts(frame, observations):
    """Draw each hand's finger count at its wrist."""
    height, width = frame.shape[:2]
    for observation in observations:
        x = int(observation.wrist_x * width)
        y = int(observation.wrist_y * height)
        draw_styled_text(
            frame,
            str(observation.finger_count),
            (x, max(34, y - 20)),
            (130, 255, 190),
            1.15,
            2,
        )


def draw_exit_hint(frame):
    """Draw the keyboard shortcut used to close the preview."""
    height, width = frame.shape[:2]
    draw_styled_text(
        frame, "Q  EXIT", (width - 115, height - 24), (190, 205, 230), 0.58, 1
    )
