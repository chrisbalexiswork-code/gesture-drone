import numpy as np
import cv2
import mediapipe as mp

# --- Kalman Filter ---
class KalmanFilter1D:
    def __init__(self):
        self.x = np.array([[1.0], [0.0]])
        self.P = np.eye(2) * 1.0
        self.F = np.array([[1, 0.05], [0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.eye(2) * 0.01
        self.R = np.array([[0.5]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0]

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K * y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0, 0]

# --- PID Controller ---
class PIDController:
    def __init__(self, kp=8.0, ki=0.2, kd=3.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.05

    def compute(self, target, current):
        error = target - current
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -2.0, 2.0)
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, -10.0, 10.0)

# --- Drone Physics ---
class Drone:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 1.0])  # x, y, z
        self.vel = np.array([0.0, 0.0, 0.0])
        self.mass = 1.0
        self.drag = 0.85
        self.dt = 0.05
        self.gravity = -9.8

    def step(self, thrust, fx=0.0):
        gravity_force = self.mass * self.gravity
        az = (thrust + gravity_force) / self.mass
        ax = fx / self.mass
        acc = np.array([ax, 0.0, az])
        self.vel = self.vel * self.drag + acc * self.dt
        self.pos = self.pos + self.vel * self.dt
        self.pos[2] = max(0.0, self.pos[2])

    def get_altitude(self):
        return self.pos[2]

# --- Gesture Detection ---
mp_hands = mp.solutions.hands

GESTURE_COLORS = {
    "Pointing_Up": (0, 255, 100),
    "Open_Palm":   (0, 200, 255),
    "Fist":        (0, 60, 255),
    "Peace":       (255, 200, 0),
    "Other":       (160, 160, 160),
    "None":        (100, 100, 100),
}

GESTURE_LABELS = {
    "Pointing_Up": "ASCEND",
    "Open_Palm":   "HOVER",
    "Fist":        "LAND",
    "Peace":       "FORWARD",
    "Other":       "IDLE",
    "None":        "NO HAND",
}

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
    elif n == 2 and fingers_up[0] and fingers_up[1]:
        return "Peace"
    return "Other"

def gesture_to_commands(gesture, current_target):
    fx = 0.0
    if gesture == "Pointing_Up":
        target = min(current_target + 0.2, 5.0)
    elif gesture == "Fist":
        target = max(current_target - 0.1, 0.0)
    elif gesture == "Peace":
        target = current_target
        fx = 8.0
    elif gesture == "Open_Palm":
        target = current_target
    else:
        target = current_target
    return target, fx

# --- Graph helpers ---
def draw_panel(frame, rect, title, color=(40,40,40)):
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (90,90,90), 1)
    cv2.putText(frame, title, (x+6, y+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

def draw_altitude_graph(frame, alt_h, target_h, kalman_h, rect):
    x, y, w, h = rect
    max_val = 5.0
    draw_panel(frame, rect, "ALTITUDE CONTROL (m)")

    for level in [1,2,3,4,5]:
        gy = y + h - int((level / max_val) * h)
        cv2.line(frame, (x, gy), (x+w, gy), (55,55,55), 1)
        cv2.putText(frame, f"{level}", (x+2, gy-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (110,110,110), 1)

    n = len(alt_h)
    if n < 2:
        return
    window = min(n, w)
    for i in range(1, window):
        x1 = x + int((i-1)*w/window)
        x2 = x + int(i*w/window)
        # Target dashed red
        ty1 = y+h - int((target_h[-(window-i+1)] / max_val)*h)
        ty2 = y+h - int((target_h[-(window-i)]   / max_val)*h)
        if i % 6 < 3:
            cv2.line(frame, (x1,ty1),(x2,ty2),(0,0,200),1)
        # Kalman green
        ky1 = y+h - int((kalman_h[-(window-i+1)] / max_val)*h)
        ky2 = y+h - int((kalman_h[-(window-i)]   / max_val)*h)
        cv2.line(frame, (x1,ky1),(x2,ky2),(0,180,80),1)
        # Actual orange
        ay1 = y+h - int((alt_h[-(window-i+1)] / max_val)*h)
        ay2 = y+h - int((alt_h[-(window-i)]   / max_val)*h)
        cv2.line(frame, (x1,ay1),(x2,ay2),(0,140,255),2)

    # Legend
    for i,(label,color) in enumerate([("Actual",(0,140,255)),("Target",(0,0,200)),("Kalman",(0,180,80))]):
        cv2.putText(frame, label, (x+w-110+i*37, y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, color, 1)

    # Live value dot
    if alt_h:
        dot_y = y+h - int((alt_h[-1]/max_val)*h)
        cv2.circle(frame, (x+w-3, dot_y), 4, (0,140,255), -1)
        cv2.putText(frame, f"{alt_h[-1]:.2f}m", (x+w-48, dot_y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,140,255), 1)

def draw_error_graph(frame, alt_h, target_h, rect):
    x, y, w, h = rect
    max_err = 3.0
    mid = y + h//2
    draw_panel(frame, rect, "PID TRACKING ERROR (m)")

    for level in [-2,-1,0,1,2]:
        gy = mid - int((level/max_err)*(h//2))
        cv2.line(frame, (x,gy),(x+w,gy),(55,55,55),1)
        cv2.putText(frame, f"{level:+d}", (x+2,gy-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100,100,100), 1)
    cv2.line(frame,(x,mid),(x+w,mid),(90,90,90),1)

    n = len(alt_h)
    if n < 2:
        return
    window = min(n, w)
    errors = [target_h[i] - alt_h[i] for i in range(len(alt_h))]
    for i in range(1, window):
        x1 = x + int((i-1)*w/window)
        x2 = x + int(i*w/window)
        e1 = errors[-(window-i+1)]
        e2 = errors[-(window-i)]
        y1 = mid - int((e1/max_err)*(h//2))
        y2 = mid - int((e2/max_err)*(h//2))
        color = (0,220,80) if abs(e2) < 0.2 else (0,80,255)
        cv2.line(frame,(x1,y1),(x2,y2),color,2)

    if errors:
        cv2.putText(frame, f"err:{errors[-1]:+.2f}m", (x+w-70,y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,220,200), 1)

def draw_forward_graph(frame, x_h, rect):
    x, y, w, h = rect
    max_x = 10.0
    mid = y + h//2
    draw_panel(frame, rect, "FORWARD / BACK (m)")

    for level in [-8,-4,0,4,8]:
        gy = mid - int((level/max_x)*(h//2))
        cv2.line(frame,(x,gy),(x+w,gy),(55,55,55),1)
        cv2.putText(frame, f"{level}", (x+2,gy-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100,100,100), 1)
    cv2.line(frame,(x,mid),(x+w,mid),(90,90,90),1)

    n = len(x_h)
    if n < 2:
        return
    window = min(n, w)
    for i in range(1, window):
        x1 = x + int((i-1)*w/window)
        x2 = x + int(i*w/window)
        xv1 = x_h[-(window-i+1)]
        xv2 = x_h[-(window-i)]
        y1 = mid - int((xv1/max_x)*(h//2))
        y2 = mid - int((xv2/max_x)*(h//2))
        cv2.line(frame,(x1,y1),(x2,y2),(255,180,0),2)

    if x_h:
        cv2.putText(frame, f"{x_h[-1]:.2f}m", (x+w-55,y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,180,0), 1)

# --- Main ---
def main():
    drone = Drone()
    kalman = KalmanFilter1D()
    pid = PIDController()

    target_altitude = 1.0
    gesture = "None"

    alt_history    = []
    target_history = []
    kalman_history = []
    x_history      = []

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            fx = 0.0
            if results.multi_hand_landmarks:
                for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                    gesture = get_gesture(lm, hd)
                    target_altitude, fx = gesture_to_commands(gesture, target_altitude)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, lm, mp_hands.HAND_CONNECTIONS)

            raw_alt = drone.get_altitude()
            kalman.predict()
            smooth_alt = kalman.update(raw_alt)
            thrust = pid.compute(target_altitude, smooth_alt)
            drone.step(thrust, fx)

            alt = drone.get_altitude()
            alt_history.append(alt)
            target_history.append(target_altitude)
            kalman_history.append(smooth_alt)
            x_history.append(drone.pos[0])

            # --- Graphs ---
            gh = 130  # graph height
            gw = w//3 - 15
            gy = h - gh - 8

            draw_altitude_graph(frame, alt_history, target_history, kalman_history,
                                (8, gy, gw, gh))
            draw_error_graph(frame, alt_history, target_history,
                             (gw+16, gy, gw, gh))
            draw_forward_graph(frame, x_history,
                               (gw*2+24, gy, gw, gh))

            # --- TOP RIGHT: Gesture Box ---
            g_color = GESTURE_COLORS.get(gesture, (160,160,160))
            g_label = GESTURE_LABELS.get(gesture, "IDLE")
            box_w, box_h = 220, 60
            bx = w - box_w - 10
            by = 10
            cv2.rectangle(frame, (bx, by), (bx+box_w, by+box_h), (20,20,20), -1)
            cv2.rectangle(frame, (bx, by), (bx+box_w, by+box_h), g_color, 2)
            cv2.putText(frame, "GESTURE", (bx+8, by+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
            cv2.putText(frame, g_label, (bx+8, by+48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, g_color, 2)

            # --- LEFT: Drone State Panel ---
            panel_w = 230
            panel_h = 230
            cv2.rectangle(frame, (8, 8), (8+panel_w, 8+panel_h), (20,20,20), -1)
            cv2.rectangle(frame, (8, 8), (8+panel_w, 8+panel_h), (70,70,70), 1)
            cv2.putText(frame, "DRONE STATE", (14, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

            state_lines = [
                (f"X:      {drone.pos[0]:+7.2f} m", (255,200,0)),
                (f"Y:      {drone.pos[1]:+7.2f} m", (255,200,0)),
                (f"Z(alt): {drone.pos[2]:+7.2f} m", (0,140,255)),
                (f"Vx:     {drone.vel[0]:+7.2f} m/s",(200,160,0)),
                (f"Vz:     {drone.vel[2]:+7.2f} m/s",(0,100,200)),
                (f"Target: {target_altitude:+7.2f} m", (0,0,200)),
                (f"Kalman: {smooth_alt:+7.2f} m",   (0,180,80)),
                (f"Thrust: {thrust:+7.2f} N",        (200,0,200)),
            ]
            for i, (text, color) in enumerate(state_lines):
                cv2.putText(frame, text, (14, 50 + i*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

            # Coordinates box bottom left
            coord_text = f"POS ({drone.pos[0]:.1f}, {drone.pos[1]:.1f}, {drone.pos[2]:.1f})"
            cv2.putText(frame, coord_text, (10, gy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,220,255), 1)

            # Controls hint
            cv2.putText(frame,
                        "UP=ascend | FIST=land | PALM=hover | PEACE=forward | Q=quit",
                        (250, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140,140,140), 1)

            cv2.imshow("Gesture Drone | Kalman + PID", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()