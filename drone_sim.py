import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrow
import cv2
import mediapipe as mp

# --- Drone Physics ---
class Drone:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 1.0])  # x, y, z
        self.vel = np.array([0.0, 0.0, 0.0])
        self.mass = 1.0
        self.drag = 0.8
        self.dt = 0.05

    def apply_command(self, command):
        force = np.array([0.0, 0.0, 0.0])
        gravity = np.array([0.0, 0.0, -2.0])

        if command == "UP":
            force = np.array([0.0, 0.0, 5.0])
        elif command == "DOWN":
            force = np.array([0.0, 0.0, -1.0])
        elif command == "FORWARD":
            force = np.array([3.0, 0.0, 2.0])
        elif command == "HOVER":
            force = np.array([0.0, 0.0, 2.0])  # counteract gravity
        elif command == "LAND":
            force = np.array([0.0, 0.0, -5.0])

        total_force = force + gravity
        acc = total_force / self.mass
        self.vel = self.vel * self.drag + acc * self.dt
        self.pos = self.pos + self.vel * self.dt
        self.pos[2] = max(0.0, self.pos[2])  # no underground

    def get_state(self):
        return self.pos.copy()

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

def gesture_to_command(gesture):
    mapping = {
        "Pointing_Up": "UP",
        "Fist":        "LAND",
        "Open_Palm":   "HOVER",
        "Peace":       "FORWARD",
        "Other":       "HOVER"
    }
    return mapping.get(gesture, "HOVER")

# --- Main ---
def main():
    drone = Drone()
    trajectory = [drone.get_state().copy()]

    cap = cv2.VideoCapture(0)

    plt.ion()
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    command = "HOVER"
    gesture = "None"

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
                    command = gesture_to_command(gesture)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, lm, mp_hands.HAND_CONNECTIONS)

            drone.apply_command(command)
            pos = drone.get_state()
            trajectory.append(pos.copy())

            # HUD
            cv2.putText(frame, f"Gesture: {gesture}",  (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Command: {command}",  (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(frame, f"Alt: {pos[2]:.2f}m",  (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.putText(frame, "Q=quit", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
            cv2.imshow("Drone Control", frame)

            # Plot
            traj = np.array(trajectory)
            ax1.cla()
            ax1.plot(traj[:,0], traj[:,1], traj[:,2], 'b-', linewidth=1)
            ax1.scatter(*pos, color='red', s=100, zorder=5)
            ax1.set_xlim(-5, 5); ax1.set_ylim(-5, 5); ax1.set_zlim(0, 5)
            ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z (Alt)')
            ax1.set_title('Drone Trajectory')

            ax2.cla()
            ax2.plot(traj[:,2], 'g-')
            ax2.set_ylim(0, 5)
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Altitude (m)')
            ax2.set_title(f'Altitude | Cmd: {command}')
            ax2.axhline(y=pos[2], color='r', linestyle='--', alpha=0.5)

            plt.pause(0.01)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()