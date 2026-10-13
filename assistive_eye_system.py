"""
assistive_eye_system.py
-----------------------
Assistive Eye-Tracking Communication & Interaction Suite.
Includes:
1. Gaze-based Mouse Control (cursor tracking, wink clicks, head-pitch scroll, head-yaw navigation)
2. Assistive On-Screen Eye-Gaze Keyboard with Dwell Time & Instant Wink typing
3. Real-Time Prefix Word Prediction & Autocomplete
4. Non-Blocking Text-to-Speech (TTS) Engine for hands-free vocalization
5. Seamless Mode Switching between Free Cursor and Virtual Keyboard
"""

import cv2
import mediapipe as mp
import pyautogui as pg
import time
import numpy as np
from collections import deque
from gaze_keyboard_ui import GazeKeyboard, speak_text_async

# -------- CONFIGURATION --------
pg.PAUSE = 0
pg.FAILSAFE = True
TARGET_FPS = 30
PROCESS_SCALE = 0.60
screen_w, screen_h = pg.size()

# Dwell and Keyboard Config
KEYBOARD_DWELL_TIME = 0.75     # Seconds of gaze fixation to type a key
EYE_OPEN_GRACE = 0.15

# Click / Wink Config
WINK_CONFIRM_FRAMES = 2
WINK_RESET_FRAMES = 2
MIN_ASYMMETRY = 0.006
LEFT_COOLDOWN = 0.40
RIGHT_COOLDOWN = 0.40
WINK_RATIO = 0.72
OPEN_RATIO = 0.78
WINK_START_RATIO = 0.96
DOUBLE_WINK_GAP = 0.55
DRAG_HOLD_FRAMES = 12
WINK_FREEZE_FRAMES = 15
CLICK_MOTION_BLOCK = 0.45

# Head Pose Navigation Config
SCROLL_HOLD_FRAMES = 12
SCROLL_COOLDOWN = 0.30
SCROLL_AMOUNT = 120
HEAD_SCROLL_UP_RATIO = 0.91
HEAD_SCROLL_DOWN_RATIO = 1.09
HEAD_YAW_LEFT_RATIO = 0.84
HEAD_YAW_RIGHT_RATIO = 1.16
HEAD_YAW_HOLD_FRAMES = 12
HEAD_YAW_COOLDOWN = 1.2

# Dynamic EAR Config
DYNAMIC_BASELINE_WINDOW = 90
DYNAMIC_UPDATE_GUARD = 0.80
DYNAMIC_ALPHA = 0.02

# Calibration Points for Gaze
CALIBRATION_SECONDS = 1.4
CURSOR_RANGE_SCALE = 0.72
CALIBRATION_POINTS = [
    ("CENTER", "Look at the center"),
    ("UP",     "Look at the top center"),
    ("DOWN",   "Look at the bottom center"),
    ("LEFT",   "Look at the left side"),
    ("RIGHT",  "Look at the right side"),
]

# Initialize MediaPipe FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRIC UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def euclidean(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_ear_6(landmarks, eye_side):
    if eye_side == "LEFT":
        p1, p2, p3 = landmarks[33], landmarks[160], landmarks[158]
        p4, p5, p6 = landmarks[133], landmarks[153], landmarks[144]
    else:
        p1, p2, p3 = landmarks[362], landmarks[385], landmarks[387]
        p4, p5, p6 = landmarks[263], landmarks[373], landmarks[380]
    A = euclidean(p2, p6)
    B = euclidean(p3, p5)
    C = euclidean(p1, p4)
    return (A + B) / (2.0 * C + 1e-6)

def get_head_pitch_metric(landmarks):
    eye_span = euclidean(landmarks[33], landmarks[263])
    return (landmarks[152].y - landmarks[1].y) / max(eye_span, 1e-6)

def get_head_yaw_metric(landmarks):
    nose_x = landmarks[1].x
    left_ear_x = landmarks[234].x
    right_ear_x = landmarks[454].x
    span = right_ear_x - left_ear_x
    if abs(span) < 1e-4:
        return 0.5
    return float(np.clip((nose_x - left_ear_x) / span, 0.0, 1.0))

def draw_target(frame, label):
    h, w = frame.shape[:2]
    pos = {
        "CENTER": (w // 2, h // 2),
        "UP": (w // 2, h // 5),
        "DOWN": (w // 2, (4 * h) // 5),
        "LEFT": (w // 5, h // 2),
        "RIGHT": ((4 * w) // 5, h // 2)
    }
    x, y = pos[label]
    cv2.circle(frame, (x, y), 18, (0, 255, 255), 3)
    cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

def process_face_landmarks(frame):
    small = cv2.resize(frame, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)
    return face_mesh.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
def run_gaze_calibration(cam, fm):
    print("\n[Calibration]: Follow the dot with your eyes only.\n")
    samples = {name: [] for name, _ in CALIBRATION_POINTS}
    for name, instruction in CALIBRATION_POINTS:
        start = time.time()
        while time.time() - start < CALIBRATION_SECONDS:
            ret, frame = cam.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            output = process_face_landmarks(frame)
            if output.multi_face_landmarks:
                iris = output.multi_face_landmarks[0].landmark[468]
                samples[name].append((iris.x, iris.y))
            remaining = max(0.0, CALIBRATION_SECONDS - (time.time() - start))
            draw_target(frame, name)
            cv2.putText(frame, "Gaze Calibration", (25, 40), 1, 1.5, (0, 255, 0), 2)
            cv2.putText(frame, instruction, (25, 80), 1, 1.4, (255, 255, 255), 2)
            cv2.putText(frame, f"{remaining:.1f}s", (25, 118), 1, 1.3, (0, 255, 255), 2)
            cv2.imshow("Assistive Eye System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt
    
    defaults = {"CENTER": (0.50, 0.50), "UP": (0.50, 0.38),
                "DOWN": (0.50, 0.62), "LEFT": (0.40, 0.50), "RIGHT": (0.60, 0.50)}
    calibrated = {}
    for name, _ in CALIBRATION_POINTS:
        pts = samples[name]
        calibrated[name] = (float(np.median([p[0] for p in pts])),
                            float(np.median([p[1] for p in pts]))) if pts else defaults[name]

    cx, cy = calibrated["CENTER"]
    lx, _ = calibrated["LEFT"]
    rx, _ = calibrated["RIGHT"]
    _, uy = calibrated["UP"]
    _, dy = calibrated["DOWN"]
    x_span = max(abs(rx - lx), 0.08)
    y_span = max(abs(dy - uy), 0.08)
    xm = max(0.02, x_span * (1.0 - CURSOR_RANGE_SCALE) * 0.5)
    yhr = max(0.04, (y_span * CURSOR_RANGE_SCALE) / 2.0)
    swh = max(0.04, x_span * 0.18)

    config = {
        "gaze_x_min": float(np.clip(min(lx, rx) - xm, 0, 1)),
        "gaze_x_max": float(np.clip(max(lx, rx) + xm, 0, 1)),
        "gaze_y_min": float(np.clip(cy - yhr, 0, 1)),
        "gaze_y_max": float(np.clip(cy + yhr, 0, 1)),
        "scroll_zone_min_x": float(np.clip(cx - swh, 0, 1)),
        "scroll_zone_max_x": float(np.clip(cx + swh, 0, 1)),
        "scroll_top_y": float(np.clip((cy + uy) / 2, 0, 1)),
        "scroll_bottom_y": float(np.clip((cy + dy) / 2, 0, 1)),
    }
    print("[Calibration Complete]: Gaze range configured successfully.\n")
    return config

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ASSISTIVE RUNTIME
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" ASSISTIVE EYE-TRACKING & VIRTUAL KEYBOARD SYSTEM")
    print(" Mode 1: Free Cursor & Wink/Head Controls")
    print(" Mode 2: On-Screen Eye Keyboard with Word Prediction & Speech")
    print(" Press 'K' to toggle Keyboard Mode  |  Press 'Q' to quit")
    print("=" * 60)

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # 1. Baseline EAR & Head Calibration
    print("\nCalibrating EAR baseline — Keep both eyes open (3s)...")
    calib_ear_left, calib_ear_right = [], []
    calib_head_pitch, calib_head_yaw = [], []

    start = time.time()
    while time.time() - start < 3:
        ret, frame = camera.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        res = process_face_landmarks(frame)
        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            calib_ear_left.append(get_ear_6(lms, "LEFT"))
            calib_ear_right.append(get_ear_6(lms, "RIGHT"))
            calib_head_pitch.append(get_head_pitch_metric(lms))
            calib_head_yaw.append(get_head_yaw_metric(lms))
        remaining = int(3 - (time.time() - start)) + 1
        cv2.putText(frame, f"Calibrating EAR... {remaining}s", (30, 50), 1, 2, (0, 255, 0), 2)
        cv2.imshow("Assistive Eye System", frame)
        cv2.waitKey(1)

    left_baseline_live = float(np.mean(calib_ear_left)) if calib_ear_left else 0.25
    right_baseline_live = float(np.mean(calib_ear_right)) if calib_ear_right else 0.25
    head_pitch_baseline = float(np.mean(calib_head_pitch)) if calib_head_pitch else 1.0
    head_yaw_baseline = float(np.mean(calib_head_yaw)) if calib_head_yaw else 0.5

    left_ear_window = deque(maxlen=DYNAMIC_BASELINE_WINDOW)
    right_ear_window = deque(maxlen=DYNAMIC_BASELINE_WINDOW)
    for v in calib_ear_left[-DYNAMIC_BASELINE_WINDOW:]:
        left_ear_window.append(v)
    for v in calib_ear_right[-DYNAMIC_BASELINE_WINDOW:]:
        right_ear_window.append(v)

    HEAD_SCROLL_UP_THR = head_pitch_baseline * HEAD_SCROLL_UP_RATIO
    HEAD_SCROLL_DOWN_THR = head_pitch_baseline * HEAD_SCROLL_DOWN_RATIO
    HEAD_YAW_LEFT_THR = head_yaw_baseline * HEAD_YAW_LEFT_RATIO
    HEAD_YAW_RIGHT_THR = head_yaw_baseline * HEAD_YAW_RIGHT_RATIO

    # 2. Gaze Calibration
    gaze_config = run_gaze_calibration(camera, face_mesh)

    # 3. System States
    current_mode = "KEYBOARD"   # "MOUSE" or "KEYBOARD"
    keyboard_ui = GazeKeyboard(screen_w=screen_w, screen_h=screen_h, dwell_time=KEYBOARD_DWELL_TIME)

    smooth_x, smooth_y = screen_w // 2, screen_h // 2
    left_wink_frames = right_wink_frames = 0
    left_reset_frames = right_reset_frames = 0
    left_wink_active_frames = right_wink_active_frames = 0
    left_wink_start_time = 0.0
    last_left_click_time = 0.0
    last_right_click_time = 0.0
    last_scroll_time = 0.0
    last_yaw_action_time = 0.0
    drag_mode = False
    is_clicking = False
    is_scrolling = False
    motion_blocked_until = 0.0
    eyes_closed_time = 0.0
    pitch_up_count = pitch_down_count = 0
    yaw_left_count = yaw_right_count = 0

    cv2.namedWindow("Assistive Eye System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Assistive Eye System", screen_w, screen_h)

    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        now = time.time()

        res = process_face_landmarks(frame)
        wink_confirmed = False

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            iris = lms[468]

            # Dynamic EAR calculation
            left_ear = get_ear_6(lms, "LEFT")
            right_ear = get_ear_6(lms, "RIGHT")
            head_pitch = get_head_pitch_metric(lms)
            head_yaw = get_head_yaw_metric(lms)

            # Update live baseline if eyes are open
            if (left_ear > left_baseline_live * DYNAMIC_UPDATE_GUARD and
                right_ear > right_baseline_live * DYNAMIC_UPDATE_GUARD):
                left_ear_window.append(left_ear)
                right_ear_window.append(right_ear)
                if len(left_ear_window) >= 10:
                    med_l = float(np.median(left_ear_window))
                    med_r = float(np.median(right_ear_window))
                    left_baseline_live = (1.0 - DYNAMIC_ALPHA) * left_baseline_live + DYNAMIC_ALPHA * med_l
                    right_baseline_live = (1.0 - DYNAMIC_ALPHA) * right_baseline_live + DYNAMIC_ALPHA * med_r

            L_WINK_THR = left_baseline_live * WINK_RATIO
            R_WINK_THR = right_baseline_live * WINK_RATIO
            L_OPEN_THR = left_baseline_live * OPEN_RATIO
            R_OPEN_THR = right_baseline_live * OPEN_RATIO

            # Wink Detection
            left_candidate = (left_ear < L_WINK_THR and right_ear > R_OPEN_THR and
                              (right_ear - left_ear) > MIN_ASYMMETRY)
            right_candidate = (right_ear < R_WINK_THR and left_ear > L_OPEN_THR and
                               (left_ear - right_ear) > MIN_ASYMMETRY)

            # Left wink progression
            if left_candidate:
                left_wink_frames += 1
                left_reset_frames = 0
                if left_wink_frames == 1:
                    left_wink_start_time = now
            else:
                left_reset_frames += 1
                if left_reset_frames >= WINK_RESET_FRAMES:
                    if left_wink_frames >= WINK_CONFIRM_FRAMES and (now - last_left_click_time) > LEFT_COOLDOWN:
                        wink_confirmed = True
                        left_wink_active_frames = WINK_FREEZE_FRAMES
                        last_left_click_time = now
                        motion_blocked_until = now + CLICK_MOTION_BLOCK
                    left_wink_frames = 0

            # Right wink progression
            if right_candidate:
                right_wink_frames += 1
                right_reset_frames = 0
            else:
                right_reset_frames += 1
                if right_reset_frames >= WINK_RESET_FRAMES:
                    if right_wink_frames >= WINK_CONFIRM_FRAMES and (now - last_right_click_time) > RIGHT_COOLDOWN:
                        right_wink_active_frames = WINK_FREEZE_FRAMES
                        last_right_click_time = now
                        motion_blocked_until = now + CLICK_MOTION_BLOCK
                        if current_mode == "MOUSE":
                            pg.click(button='right')
                    right_wink_frames = 0

            # Gaze cursor mapping
            xr = max(gaze_config["gaze_x_max"] - gaze_config["gaze_x_min"], 1e-6)
            yr = max(gaze_config["gaze_y_max"] - gaze_config["gaze_y_min"], 1e-6)
            norm_x = (iris.x - gaze_config["gaze_x_min"]) / xr
            norm_y = (iris.y - gaze_config["gaze_y_min"]) / yr

            tx = float(np.clip(norm_x, 0.0, 1.0)) * screen_w
            ty = float(np.clip(norm_y, 0.0, 1.0)) * screen_h

            dist = np.hypot(tx - smooth_x, ty - smooth_y)
            blend = 0.14 if dist > 200 else (0.08 if dist > 80 else 0.035)
            smooth_x = smooth_x + (tx - smooth_x) * blend
            smooth_y = smooth_y + (ty - smooth_y) * blend

            cx = int(np.clip(smooth_x, 20, screen_w - 20))
            cy = int(np.clip(smooth_y, 20, screen_h - 20))

            # ─────────────────────────────────────────────────────────────
            # MODE BRANCHING
            # ─────────────────────────────────────────────────────────────
            if current_mode == "MOUSE":
                # Free cursor movement
                if now > motion_blocked_until:
                    pg.moveTo(cx, cy)

                # Mouse clicks
                if wink_confirmed:
                    pg.click(button='left')

                # Head Scroll Navigation
                if head_pitch < HEAD_SCROLL_UP_THR:
                    pitch_up_count += 1
                    if pitch_up_count >= SCROLL_HOLD_FRAMES and (now - last_scroll_time) > SCROLL_COOLDOWN:
                        pg.scroll(SCROLL_AMOUNT)
                        last_scroll_time = now
                elif head_pitch > HEAD_SCROLL_DOWN_THR:
                    pitch_down_count += 1
                    if pitch_down_count >= SCROLL_HOLD_FRAMES and (now - last_scroll_time) > SCROLL_COOLDOWN:
                        pg.scroll(-SCROLL_AMOUNT)
                        last_scroll_time = now
                else:
                    pitch_up_count = pitch_down_count = 0

                # Render Mouse Mode Camera HUD
                hud_frame = cv2.resize(frame, (screen_w, screen_h))
                cv2.putText(hud_frame, "MODE: FREE MOUSE (Press 'K' or look at switch button)",
                            (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 120), 2)
                cv2.putText(hud_frame, f"Gaze: ({cx}, {cy}) | Pitch: {head_pitch:.2f} | Yaw: {head_yaw:.2f}",
                            (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Mode switch button on HUD
                btn_w, btn_h = 240, 60
                bx, by = screen_w - btn_w - 40, 30
                cv2.rectangle(hud_frame, (bx, by), (bx + btn_w, by + btn_h), (0, 140, 255), -1)
                cv2.putText(hud_frame, "OPEN KEYBOARD", (bx + 15, by + 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

                if bx <= cx <= bx + btn_w and by <= cy <= by + btn_h:
                    if wink_confirmed:
                        current_mode = "KEYBOARD"

                cv2.circle(hud_frame, (cx, cy), 18, (0, 255, 255), 2)
                cv2.imshow("Assistive Eye System", hud_frame)

            else:
                # KEYBOARD MODE
                # Create dark canvas for keyboard UI
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

                # Feed gaze coordinates into keyboard UI
                triggered, triggered_key = keyboard_ui.update_gaze(cx, cy, wink_confirm=wink_confirmed)
                if triggered:
                    print(f"[Key Triggered]: {triggered_key}")
                    if triggered_key == "EXIT TO MOUSE":
                        current_mode = "MOUSE"

                # Render keyboard and gaze reticle
                rendered = keyboard_ui.render(canvas, gaze_pt=(cx, cy))

                # Picture-In-Picture (PIP) camera preview in top-right corner
                pip_w, pip_h = int(screen_w * 0.18), int(screen_h * 0.18)
                small_cam = cv2.resize(frame, (pip_w, pip_h))
                cv2.rectangle(small_cam, (0, 0), (pip_w, pip_h), (0, 200, 255), 2)
                # Position PIP in the top right
                px = screen_w - pip_w - 40
                py = 55
                rendered[py:py+pip_h, px:px+pip_w] = small_cam

                cv2.imshow("Assistive Eye System", rendered)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('k'):
            current_mode = "MOUSE" if current_mode == "KEYBOARD" else "KEYBOARD"
            print(f"[Mode Switched]: {current_mode}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
