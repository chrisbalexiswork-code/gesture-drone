import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp

# --- Kalman Filter ---
class KalmanFilter1D:
    """
    1D Kalman filter for smoothing noisy altitude commands.
    Models drone altitude as a state with position and velocity.
    """
    def __init__(self):
        self.x = np.array([[1.0], [0.0]])  # state: [altitude, velocity]
        self.P = np.eye(2) * 1.0           # error covariance
        self.F = np.array([[1, 0.05],      # state transition (dt=0.05)
                           [0, 1.0]])
        self.H = np.array([[1.0, 0.0]])    # measurement matrix
        self.Q = np.eye(2) * 0.01          # process noise
        self.R = np.array([[0.5]])         # measurement noise

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
    """
    PID controller for altitude hold.
    Computes thrust force to reach target altitude.
    """
    def __init__(self, kp=4.0, ki=0.1, kd=2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.05

    def compute(self, target, current):
        error = target - current
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -2.0, 2.0)  # anti-windup
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, -10.0, 10.0)

# --- Drone Physics ---
class Drone:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 1.0])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.mass = 1.0
        self.drag = 0.85
        self.dt = 0.05
        self.gravity = -9.8

    def step(self, thrust):
        gravity_force = self.mass * self.gravity
        net_force = thrust + gravity_force
        acc = np.array([0.0, 0.0, net_force / self.mass])
        self.vel = self.vel * self.drag + acc * self.dt
        self.pos = self.pos + self.vel * self.dt
        self.pos[2] = max(0.0, self.pos[2])

    def get_altitude(self):
        return self.pos[2]

# --- Gesture Detection ---
mp_hands = mp.solutions.hands

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

def gesture_to_target(gesture, current_target):
    if gesture == "Pointing_Up":
        return min(current_target + 0.1, 5.0)    # ascend
    elif gesture == "Fist":
        return max(current_target - 0.1, 0.0)    # gradual land
    elif gesture == "Peace":
        return current_target                     # hold altitude, forward only
    elif gesture == "Open_Palm":
        return current_target                     # hover
    return current_target

# --- Main ---
def main():
    drone = Drone()
    kalman = KalmanFilter1D()
    pid = PIDController()

    target_altitude = 1.0
    gesture = "None"

    alt_history = []
    target_history = []
    kalman_history = []

    cap = cv2.VideoCapture(0)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Gesture-Controlled Drone | Kalman Filter + PID Control")

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                    gesture = get_gesture(lm, hd)
                    target_altitude = gesture_to_target(gesture, target_altitude)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, lm, mp_hands.HAND_CONNECTIONS)

            # Kalman filter on altitude measurement
            raw_alt = drone.get_altitude()
            kalman.predict()
            smooth_alt = kalman.update(raw_alt)

            # PID computes thrust to reach target
            thrust = pid.compute(target_altitude, smooth_alt)

            # Drone physics step
            drone.step(thrust)

            alt = drone.get_altitude()
            alt_history.append(alt)
            target_history.append(target_altitude)
            kalman_history.append(smooth_alt)

            # HUD
            cv2.putText(frame, f"Gesture: {gesture}",           (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Target Alt: {target_altitude:.2f}m", (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"Actual Alt: {alt:.2f}m",       (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, f"Kalman Est: {smooth_alt:.2f}m",(10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,165,0), 2)
            cv2.putText(frame, f"Thrust: {thrust:.2f}N",        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
            cv2.putText(frame, "Q=quit", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
            cv2.imshow("Drone Control", frame)

            # Plot
            window = 200
            disp_alt    = alt_history[-window:]
            disp_target = target_history[-window:]
            disp_kalman = kalman_history[-window:]

            ax1.cla()
            ax1.plot(disp_alt,    'b-',  label='Actual Alt')
            ax1.plot(disp_target, 'r--', label='Target Alt')
            ax1.plot(disp_kalman, 'g-',  label='Kalman Est', alpha=0.7)
            ax1.set_ylim(0, 6)
            ax1.set_ylabel('Altitude (m)')
            ax1.set_xlabel('Time Steps')
            ax1.set_title('Altitude Control')
            ax1.legend(loc='upper right', fontsize=8)

            ax2.cla()
            if len(alt_history) > 1:
                error = [t - a for t, a in zip(disp_target, disp_alt)]
                ax2.plot(error, 'r-', label='Tracking Error')
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax2.set_ylim(-3, 3)
                ax2.set_ylabel('Error (m)')
                ax2.set_xlabel('Time Steps')
                ax2.set_title('PID Tracking Error')
                ax2.legend(fontsize=8)

            plt.tight_layout()
            plt.pause(0.01)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()