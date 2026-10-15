from dataclasses import dataclass

import cv2
import mediapipe as mp


@dataclass
class HandObservation:
    """The hand data needed by the counter and the user interface."""

    landmarks: object
    finger_count: int
    wrist_x: float
    wrist_y: float


class HandTracker:
    """Create MediaPipe's hand tracker and convert results into observations."""

    FINGER_JOINTS = ((8, 6), (12, 10), (16, 14), (20, 18))

    def __init__(self, max_num_hands=10):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.landmark_style = self.mp_draw.DrawingSpec(
            color=(255, 190, 90), thickness=2, circle_radius=3
        )
        self.connection_style = self.mp_draw.DrawingSpec(
            color=(100, 235, 255), thickness=2, circle_radius=2
        )

    def process(self, rgb_frame):
        """Run hand detection on an RGB frame."""
        return self.hands.process(rgb_frame)

    def analyze(self, result):
        """Return one observation for each detected hand."""
        observations = []
        landmarks = result.multi_hand_landmarks or []
        handedness = result.multi_handedness or []

        for index, hand in enumerate(landmarks):
            label = handedness[index].classification[0].label
            finger_count = self._count_fingers(hand.landmark, label)
            wrist = hand.landmark[0]
            observations.append(
                HandObservation(
                    landmarks=hand,
                    finger_count=finger_count,
                    wrist_x=wrist.x,
                    wrist_y=wrist.y,
                )
            )

        return observations

    def draw_landmarks(self, frame, observations):
        """Draw styled landmarks and connections for detected hands."""
        for observation in observations:
            self.mp_draw.draw_landmarks(
                frame,
                observation.landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.landmark_style,
                self.connection_style,
            )

    def close(self):
        self.hands.close()

    def _count_fingers(self, points, handedness):
        count = sum(
            points[tip].y < points[joint].y
            for tip, joint in self.FINGER_JOINTS
        )

        if handedness == "Right":
            thumb_is_open = points[4].x < points[3].x
        else:
            thumb_is_open = points[4].x > points[3].x

        return count + int(thumb_is_open)
