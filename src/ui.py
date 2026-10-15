import cv2


NEON_PINK = (255, 0, 110)
NEON_CYAN = (0, 255, 255)
TOXIC_GREEN = (0, 255, 80)
BLOOD_RED = (0, 0, 200)
DECAY_YELLOW = (0, 255, 255)


def draw_stencil_text(frame, text, position, color=NEON_PINK, scale=1.0, thickness=2):
    """Draw distressed multi-pass lettering with a dark spray-painted edge."""
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    dark_color = tuple(max(channel - 80, 0) for channel in color)
    cv2.putText(
        frame, text, (x + 2, y + 2), font, scale, dark_color, thickness + 3,
        cv2.LINE_AA
    )
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)):
        cv2.putText(
            frame, text, (x + dx, y + dy), font, scale, color, thickness,
            cv2.LINE_AA
        )


draw_styled_text = draw_stencil_text


def draw_drips(frame, x, y_start, color, count=4, max_length=28):
    """Add small paint drips below a label."""
    height, width = frame.shape[:2]
    for index in range(count):
        offset = ((index * 17) % 19) - 9
        length = 8 + ((index * 11) % max_length)
        for step in range(length):
            y = y_start + step
            point_x = x + offset
            if 0 <= y < height and 0 <= point_x < width:
                fade = 1 - (step / max(length, 1)) * 0.65
                faded_color = tuple(int(channel * fade) for channel in color)
                cv2.circle(frame, (point_x, y), 1, faded_color, -1)


def draw_anarchy_ring(frame, center, radius=24, color=NEON_PINK):
    """Draw a compact circle-A punk marker."""
    cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)
    cv2.putText(
        frame, "A", (center[0] - 8, center[1] + 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA
    )


def draw_hud_panel(frame, total_hands, total_fingers):
    """Draw the status panel in the top-left corner."""
    panel = frame.copy()
    cv2.rectangle(panel, (16, 16), (300, 126), (4, 4, 12), -1)
    cv2.rectangle(panel, (16, 16), (300, 126), NEON_PINK, 2)
    cv2.line(panel, (32, 64), (284, 64), NEON_CYAN, 1)
    cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)

    draw_stencil_text(frame, "HAND//SCANNER", (32, 49), NEON_CYAN, 0.58, 1)
    draw_styled_text(
        frame, f"HANDS  {total_hands}", (32, 91), DECAY_YELLOW, 0.72, 2
    )
    draw_styled_text(
        frame,
        f"FINGERS  {total_fingers}",
        (32, 117),
        TOXIC_GREEN,
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
            TOXIC_GREEN,
            1.15,
            2,
        )
        draw_drips(frame, x, max(38, y - 8), TOXIC_GREEN)


def draw_exit_hint(frame):
    """Draw the keyboard shortcut used to close the preview."""
    height, width = frame.shape[:2]
    draw_styled_text(
        frame, "Q // EXIT", (width - 130, height - 24), BLOOD_RED, 0.58, 1
    )
    draw_anarchy_ring(frame, (width - 42, 45), 22, NEON_PINK)
