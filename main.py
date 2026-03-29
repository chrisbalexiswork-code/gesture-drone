import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_gesture(hand_landmarks, handedness):
    lm = hand_landmarks.landmark
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers_up = [lm[t].y < lm[p].y for t, p in zip(tips, pips)]
    
    is_right = handedness.classification[0].label == "Right"
    thumb_up = lm[4].x < lm[3].x if is_right else lm[4].x > lm[3].x
    n = sum(fingers_up)

    if n == 1 and fingers_up[0]:
        return "Pointing_Up"
    elif n >= 4:
        return "Open_Palm"
    elif n == 0 and not thumb_up:
        return "Fist"
    return "Other"

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    canvas = None
    prev_point = None
    draw_color = (0, 255, 255)
    brush = 8

    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if canvas is None:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            left_g = right_g = "None"
            draw_point = None

            if results.multi_hand_landmarks:
                for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = hd.classification[0].label
                    gesture = get_gesture(lm, hd)

                    mp_drawing.draw_landmarks(
                        frame, lm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255,255,0), thickness=2)
                    )

                    if label == "Left":
                        left_g = gesture
                    else:
                        right_g = gesture
                        tip = lm.landmark[8]
                        draw_point = (int(tip.x * w), int(tip.y * h))

            if left_g == "Open_Palm":
                canvas[:] = 0
                prev_point = None

            if right_g == "Pointing_Up" and draw_point:
                if prev_point:
                    cv2.line(canvas, prev_point, draw_point, draw_color, brush)
                prev_point = draw_point
            else:
                prev_point = None

            output = frame.copy()
            mask = canvas.any(axis=2)
            output[mask] = canvas[mask]

            status = "DRAWING" if (right_g == "Pointing_Up") else ""
            cv2.putText(output, f"Left hand: {left_g}",   (10, 30),      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),   2)
            cv2.putText(output, status,                    (w//2-60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(output, f"Right hand: {right_g}", (w-300, 30),   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),   2)
            cv2.putText(output, "Q=quit  C=clear",         (10, h-10),    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180),1)

            cv2.imshow("Air Drawing", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas[:] = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
