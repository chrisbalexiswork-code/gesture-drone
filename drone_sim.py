import numpy as np
import cv2
import mediapipe as mp
import math

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
        self.kp = kp; self.ki = ki; self.kd = kd
        self.integral = 0.0; self.prev_error = 0.0; self.dt = 0.05

    def compute(self, target, current):
        error = target - current
        self.integral = np.clip(self.integral + error * self.dt, -2.0, 2.0)
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return np.clip(self.kp*error + self.ki*self.integral + self.kd*derivative, -10.0, 10.0)

# --- Drone Physics ---
class Drone:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 1.0])  # x, y, z
        self.vel = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0  # degrees
        self.mass = 1.0; self.drag = 0.85; self.dt = 0.05; self.gravity = -9.8

    def step(self, thrust, fx=0.0, fy=0.0, yaw_rate=0.0):
        az = (thrust + self.mass * self.gravity) / self.mass
        acc = np.array([fx / self.mass, fy / self.mass, az])
        self.vel = self.vel * self.drag + acc * self.dt
        self.pos = self.pos + self.vel * self.dt
        self.pos[2] = max(0.0, self.pos[2])
        self.yaw = (self.yaw + yaw_rate * self.dt) % 360

    def get_altitude(self): return self.pos[2]

# --- Gesture ---
mp_hands = mp.solutions.hands

GESTURE_COLORS = {
    "Pointing_Up": (0,255,100),
    "Open_Palm":   (0,200,255),
    "Fist":        (0,60,255),
    "Peace":       (255,200,0),
    "Thumbs_Up":   (0,180,255),
    "Pinky":       (255,80,200),
    "Ring_Pinky":  (200,80,255),
    "Rock":        (255,100,50),
    "Other":       (130,130,130),
    "None":        (60,60,60),
}
GESTURE_LABELS = {
    "Pointing_Up": "ASCEND",
    "Open_Palm":   "HOVER",
    "Fist":        "LAND",
    "Peace":       "FORWARD",
    "Thumbs_Up":   "BACKWARD",
    "Pinky":       "LEFT",
    "Ring_Pinky":  "RIGHT",
    "Rock":        "ROTATE",
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

    idx, mid, rng, pnk = fingers_up[0], fingers_up[1], fingers_up[2], fingers_up[3]

    if n == 1 and idx:                          return "Pointing_Up"
    elif n >= 4:                                return "Open_Palm"
    elif n == 0 and not thumb_up:               return "Fist"
    elif n == 0 and thumb_up:                   return "Thumbs_Up"
    elif n == 2 and idx and mid:                return "Peace"
    elif n == 1 and pnk:                        return "Pinky"
    elif n == 2 and rng and pnk:                return "Ring_Pinky"
    elif n == 2 and idx and pnk and not mid and not rng: return "Rock"
    return "Other"

def gesture_to_commands(gesture, current_target):
    fx = fy = yaw_rate = 0.0
    if gesture == "Pointing_Up":   target = min(current_target + 0.2, 5.0)
    elif gesture == "Fist":        target = max(current_target - 0.1, 0.0)
    elif gesture == "Peace":       target = current_target; fx = 8.0
    elif gesture == "Thumbs_Up":   target = current_target; fx = -8.0
    elif gesture == "Pinky":       target = current_target; fy = -8.0
    elif gesture == "Ring_Pinky":  target = current_target; fy = 8.0
    elif gesture == "Rock":        target = current_target; yaw_rate = 60.0
    else:                          target = current_target
    return target, fx, fy, yaw_rate

# --- Drawing ---
def panel(frame, x, y, w, h, title=""):
    ov = frame.copy()
    cv2.rectangle(ov,(x,y),(x+w,y+h),(15,15,15),-1)
    cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(60,60,60),1)
    if title:
        cv2.putText(frame,title,(x+5,y+13),cv2.FONT_HERSHEY_SIMPLEX,0.33,(150,150,150),1)

def auto_scale(values, padding=0.2):
    if not values: return 0.0, 5.0
    lo=min(values); hi=max(values); rng=max(hi-lo,0.5)
    lo-=rng*padding; hi+=rng*padding
    return math.floor(lo/0.5)*0.5, math.ceil(hi/0.5)*0.5

def val_to_px(v, lo, hi, y, h):
    if hi==lo: return y+h//2
    return int(np.clip(y+h - ((v-lo)/(hi-lo))*h, y, y+h))

def draw_graph(frame, histories, colors, labels, x, y, w, h, title, center_zero=False):
    panel(frame,x,y,w,h,title)
    all_vals=[v for hist in histories for v in hist[-w:]]
    if center_zero:
        mx=max((abs(v) for v in all_vals),default=1.0); mx=max(mx,0.5)
        lo,hi=-mx*1.2,mx*1.2
    else:
        lo,hi=auto_scale(all_vals)
    for i in range(5):
        gv=lo+(hi-lo)*i/4
        gy=val_to_px(gv,lo,hi,y,h)
        cv2.line(frame,(x,gy),(x+w,gy),(40,40,40),1)
        cv2.putText(frame,f"{gv:.1f}",(x+2,gy-2),cv2.FONT_HERSHEY_SIMPLEX,0.26,(90,90,90),1)
    if center_zero:
        zy=val_to_px(0,lo,hi,y,h)
        cv2.line(frame,(x,zy),(x+w,zy),(80,80,80),1)
    for hist,color,label in zip(histories,colors,labels):
        n=len(hist)
        if n<2: continue
        window=min(n,w)
        for i in range(1,window):
            x1=x+int((i-1)*w/window); x2=x+int(i*w/window)
            y1=val_to_px(hist[-(window-i+1)],lo,hi,y,h)
            y2=val_to_px(hist[-(window-i)],lo,hi,y,h)
            cv2.line(frame,(x1,y1),(x2,y2),color,2)
    for i,(label,color) in enumerate(zip(labels,colors)):
        cv2.putText(frame,label,(x+w-len(labels)*36+i*36,y+13),
                    cv2.FONT_HERSHEY_SIMPLEX,0.27,color,1)
    if histories and histories[0]:
        live=histories[0][-1]
        ly=val_to_px(live,lo,hi,y,h)
        cv2.circle(frame,(x+w-2,ly),4,colors[0],-1)
        cv2.putText(frame,f"{live:.2f}",(x+w-44,ly-4),cv2.FONT_HERSHEY_SIMPLEX,0.3,colors[0],1)

def draw_altitude_bar(frame, alt, target, x, y, w, h):
    panel(frame,x,y,w,h,"ALT")
    max_val=5.0; bx=x+w//2-7; bw=14
    cv2.rectangle(frame,(bx,y+16),(bx+bw,y+h-4),(30,30,30),-1)
    cv2.rectangle(frame,(bx,y+16),(bx+bw,y+h-4),(55,55,55),1)
    fill_h=int((alt/max_val)*(h-20)); fill_y=y+h-4-fill_h
    color=(0,200,60) if alt>0.5 else (0,60,255)
    if fill_h>0:
        cv2.rectangle(frame,(bx,fill_y),(bx+bw,y+h-4),color,-1)
    tgt_y=y+h-4-int((target/max_val)*(h-20))
    cv2.line(frame,(bx-3,tgt_y),(bx+bw+3,tgt_y),(0,0,200),2)
    for lv in [0,1,2,3,4,5]:
        ly=y+h-4-int((lv/max_val)*(h-20))
        cv2.putText(frame,str(lv),(x+1,ly+3),cv2.FONT_HERSHEY_SIMPLEX,0.25,(80,80,80),1)
    cv2.putText(frame,f"{alt:.1f}",(x+1,y+h-6),cv2.FONT_HERSHEY_SIMPLEX,0.3,color,1)

def draw_birdseye(frame, x_h, y_h, pos, x, y, w, h):
    panel(frame,x,y,w,h,"MAP (top view)")
    cx=x+w//2; cy=y+h//2
    max_range=max(abs(pos[0])+2,abs(pos[1])+2,5.0)
    scale=min(w,h)*0.42/max_range
    for g in [-2,-1,0,1,2]:
        gx=cx+int(g*max_range*scale/2); gy2=cy+int(g*max_range*scale/2)
        cv2.line(frame,(gx,y+16),(gx,y+h),(35,35,35),1)
        cv2.line(frame,(x,gy2),(x+w,gy2),(35,35,35),1)
    cv2.line(frame,(cx,y+16),(cx,y+h),(55,55,55),1)
    cv2.line(frame,(x,cy),(x+w,cy),(55,55,55),1)
    n=len(x_h)
    if n>1:
        window=min(n,300)
        for i in range(1,window):
            px1=cx+int(x_h[-(window-i+1)]*scale)
            py1=cy-int(y_h[-(window-i+1)]*scale)
            px2=cx+int(x_h[-(window-i)]*scale)
            py2=cy-int(y_h[-(window-i)]*scale)
            a=int(60+190*(i/window))
            cv2.line(frame,
                (int(np.clip(px1,x,x+w)),int(np.clip(py1,y,y+h))),
                (int(np.clip(px2,x,x+w)),int(np.clip(py2,y,y+h))),
                (0,a,a//2),1)
    # Drone with yaw arrow
    dpx=int(np.clip(cx+pos[0]*scale,x+5,x+w-5))
    dpy=int(np.clip(cy-pos[1]*scale,y+18,y+h-5))
    cv2.circle(frame,(dpx,dpy),7,(0,255,140),-1)
    cv2.circle(frame,(dpx,dpy),9,(255,255,255),1)
    cv2.circle(frame,(cx,cy),3,(90,90,90),-1)
    cv2.putText(frame,f"({pos[0]:.1f},{pos[1]:.1f})",(x+3,y+h-3),
                cv2.FONT_HERSHEY_SIMPLEX,0.27,(0,200,140),1)

def draw_compass(frame, yaw, cx, cy, r=30):
    cv2.circle(frame,(cx,cy),r,(20,20,20),-1)
    cv2.circle(frame,(cx,cy),r,(70,70,70),1)
    angle=math.radians(yaw-90)
    ex=cx+int((r-4)*math.cos(angle))
    ey=cy+int((r-4)*math.sin(angle))
    cv2.arrowedLine(frame,(cx,cy),(ex,ey),(0,200,255),2,tipLength=0.4)
    cv2.putText(frame,"N",(cx-4,cy-r+10),cv2.FONT_HERSHEY_SIMPLEX,0.28,(150,150,150),1)
    cv2.putText(frame,f"{yaw:.0f}°",(cx-12,cy+r+12),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,200,255),1)

def draw_status(frame, drone, thrust, x, y, w):
    ov=frame.copy()
    cv2.rectangle(ov,(x,y),(x+w,y+20),(15,15,15),-1)
    cv2.addWeighted(ov,0.8,frame,0.2,0,frame)
    cv2.rectangle(frame,(x,y),(x+w,y+20),(50,50,50),1)
    armed=drone.get_altitude()>0.05
    ac=(0,220,60) if armed else (0,60,220)
    mc=(0,200,60) if abs(thrust)>0.5 else (60,60,60)
    cv2.putText(frame,"ARMED" if armed else "DISARMED",(x+5,y+14),cv2.FONT_HERSHEY_SIMPLEX,0.37,ac,1)
    cv2.putText(frame,"MOTORS",(x+80,y+14),cv2.FONT_HERSHEY_SIMPLEX,0.37,mc,1)
    cv2.putText(frame,"KALMAN ON",(x+160,y+14),cv2.FONT_HERSHEY_SIMPLEX,0.33,(0,170,90),1)
    cv2.putText(frame,"PID  KP:8.0 KI:0.2 KD:3.0",(x+260,y+14),cv2.FONT_HERSHEY_SIMPLEX,0.33,(160,100,0),1)

# --- Main ---
def main():
    drone=Drone(); kalman=KalmanFilter1D(); pid=PIDController()
    target_altitude=1.0; gesture="None"
    alt_h=[]; tgt_h=[]; klm_h=[]; x_h=[]; y_h=[]; err_h=[]; yaw_h=[]

    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    with mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret: break
            frame=cv2.flip(frame,1)
            fh,fw=frame.shape[:2]

            results=hands.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            fx=fy=yaw_rate=0.0
            if results.multi_hand_landmarks:
                for lm,hd in zip(results.multi_hand_landmarks,results.multi_handedness):
                    gesture=get_gesture(lm,hd)
                    target_altitude,fx,fy,yaw_rate=gesture_to_commands(gesture,target_altitude)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,lm,mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,255,150),thickness=2,circle_radius=3),
                        mp.solutions.drawing_utils.DrawingSpec(color=(255,255,255),thickness=1))
            else:
                gesture="None"

            raw_alt=drone.get_altitude(); kalman.predict()
            smooth_alt=kalman.update(raw_alt)
            thrust=pid.compute(target_altitude,smooth_alt)
            drone.step(thrust,fx,fy,yaw_rate)
            alt=drone.get_altitude()

            alt_h.append(alt); tgt_h.append(target_altitude)
            klm_h.append(smooth_alt); x_h.append(drone.pos[0])
            y_h.append(drone.pos[1]); err_h.append(target_altitude-alt)
            yaw_h.append(drone.yaw)

            # Layout
            gh=130; gw=fw//4-12; gy=fh-gh-22

            draw_status(frame,drone,thrust,0,fh-20,fw)

            # 4 auto-scaling graphs
            draw_graph(frame,[alt_h,tgt_h,klm_h],
                       [(0,130,255),(0,0,180),(0,160,70)],
                       ["ACT","TGT","KLM"],
                       8,gy,gw,gh,"ALTITUDE (m)")
            draw_graph(frame,[err_h],[(0,80,255)],["ERR"],
                       gw+16,gy,gw,gh,"PID ERROR (m)",center_zero=True)
            draw_graph(frame,[x_h],[(255,170,0)],["FWD/BACK"],
                       gw*2+24,gy,gw,gh,"FORWARD / BACK (m)",center_zero=True)
            draw_graph(frame,[y_h],[(255,80,200)],["L/R"],
                       gw*3+32,gy,gw,gh,"LEFT / RIGHT (m)",center_zero=True)

            # Right side
            draw_altitude_bar(frame,alt,target_altitude,fw-42,75,38,200)
            draw_birdseye(frame,x_h,y_h,drone.pos,fw-185,75,138,150)
            draw_compass(frame,drone.yaw,fw-111,255)

            # Gesture box
            gc=GESTURE_COLORS.get(gesture,(130,130,130))
            gl=GESTURE_LABELS.get(gesture,"IDLE")
            bw,bh=200,58; bx=fw-bw-195; by=8
            ov=frame.copy()
            cv2.rectangle(ov,(bx,by),(bx+bw,by+bh),(15,15,15),-1)
            cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
            cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),gc,2)
            cv2.putText(frame,"GESTURE",(bx+6,by+15),cv2.FONT_HERSHEY_SIMPLEX,0.37,(160,160,160),1)
            cv2.putText(frame,gl,(bx+6,by+48),cv2.FONT_HERSHEY_SIMPLEX,0.9,gc,2)

            # Drone telemetry panel
            pw,ph=215,240
            ov=frame.copy()
            cv2.rectangle(ov,(6,6),(6+pw,6+ph),(15,15,15),-1)
            cv2.addWeighted(ov,0.78,frame,0.22,0,frame)
            cv2.rectangle(frame,(6,6),(6+pw,6+ph),(65,65,65),1)
            cv2.putText(frame,"DRONE TELEMETRY",(11,21),cv2.FONT_HERSHEY_SIMPLEX,0.4,(170,170,170),1)
            lines=[
                (f"X:      {drone.pos[0]:+8.2f} m",(255,190,0)),
                (f"Y:      {drone.pos[1]:+8.2f} m",(255,190,0)),
                (f"Z(alt): {drone.pos[2]:+8.2f} m",(0,130,255)),
                (f"Vx:     {drone.vel[0]:+8.2f} m/s",(200,140,0)),
                (f"Vy:     {drone.vel[1]:+8.2f} m/s",(180,120,0)),
                (f"Vz:     {drone.vel[2]:+8.2f} m/s",(0,90,200)),
                (f"Yaw:    {drone.yaw:8.1f} deg",(0,200,255)),
                (f"Target: {target_altitude:+8.2f} m",(0,0,200)),
                (f"Kalman: {smooth_alt:+8.2f} m",(0,160,70)),
                (f"Thrust: {thrust:+8.2f} N",(180,0,200)),
                (f"Error:  {target_altitude-alt:+8.2f} m",(0,180,180)),
            ]
            for i,(txt,col) in enumerate(lines):
                cv2.putText(frame,txt,(11,38+i*19),cv2.FONT_HERSHEY_SIMPLEX,0.36,col,1)

            # Coordinates
            cv2.putText(frame,
                f"POS  X:{drone.pos[0]:.1f}  Y:{drone.pos[1]:.1f}  Z:{drone.pos[2]:.1f}  YAW:{drone.yaw:.0f}°",
                (8,gy-10),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,210,255),1)

            # Gesture legend
            legend=[
                ("UP=ascend","FIST=land","PALM=hover","PEACE=fwd"),
                ("THUMB=back","PINKY=left","RING+PINKY=right","ROCK=rotate"),
            ]
            for row,items in enumerate(legend):
                cv2.putText(frame,"  |  ".join(items),
                    (8,fh-24+row*(-12 if row==0 else 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.32,(100,100,100),1)
            cv2.putText(frame,"THUMB=back | PINKY=left | RING+PINKY=right | ROCK=rotate | Q=quit",
                (8,fh-12),cv2.FONT_HERSHEY_SIMPLEX,0.32,(100,100,100),1)

            cv2.imshow("Gesture Drone GCS | Kalman + PID",frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()