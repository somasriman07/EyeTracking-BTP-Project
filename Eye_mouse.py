import cv2
import mediapipe as mp
import pyautogui as pg
import time
import numpy as np
from collections import deque

# -------- CONFIGURATION --------
pg.PAUSE = 0
pg.FAILSAFE = True
alpha = 0.07
MOUTH_THRESHOLD = 0.06
SCROLL_HOLD_FRAMES = 12          # back to original — easier to scroll
SCROLL_COOLDOWN = 0.30
SCROLL_AMOUNT = 120
HEAD_SCROLL_UP_RATIO = 0.91      # easier — small chin up triggers scroll
HEAD_SCROLL_DOWN_RATIO = 1.09    # easier — small chin down triggers scroll
CALIBRATION_SECONDS = 1.4
CURSOR_RANGE_SCALE = 0.72
MAX_CURSOR_STEP = 35             # allow fast sweeps
CURSOR_RESPONSE_EXPONENT = 1.6   # more linear — easier to predict
CURSOR_DEADZONE = 0.048          # filters tremor near small buttons
PRECISION_ZONE = 0.18
PRECISION_ALPHA = 0.10           # snappier precision
FAST_ALPHA = 0.20                # faster response when moving far
PRECISION_STEP = 12
TARGET_FPS = 20
PROCESS_SCALE = 0.60
EYE_OPEN_GRACE = 0.15            # was 0.35 — pause cursor sooner after blink
CALIBRATION_POINTS = [
    ("CENTER", "Look at the center"),
    ("UP",     "Look at the top center"),
    ("DOWN",   "Look at the bottom center"),
    ("LEFT",   "Look at the left side"),
    ("RIGHT",  "Look at the right side"),
]

# ── Click Accuracy Config ──────────────────────────────────────────────────
WINK_CONFIRM_FRAMES = 2          # was 3 — detect faster winks
WINK_RESET_FRAMES   = 2
MIN_ASYMMETRY       = 0.006      # was 0.008 — catch subtler winks
LEFT_COOLDOWN       = 0.40       # was 0.55 — less wait between clicks
RIGHT_COOLDOWN      = 0.40       # was 0.55
WINK_RATIO          = 0.72       # was 0.68 — easier to trigger wink
OPEN_RATIO          = 0.78       # was 0.82 — more forgiving open eye check
WINK_START_RATIO    = 0.96
DOUBLE_WINK_GAP     = 0.55
DRAG_HOLD_FRAMES    = 12

# ── Cursor freeze ─────────────────────────────────────────────────────────
WINK_FREEZE_FRAMES  = 15         # was 12 — freeze longer after click
CLICK_MOTION_BLOCK  = 0.45       # was 0.35 — block cursor longer after click

# ── Dynamic EAR Baseline Config ───────────────────────────────────────────
DYNAMIC_BASELINE_WINDOW = 90
DYNAMIC_UPDATE_GUARD    = 0.80
DYNAMIC_ALPHA           = 0.02

# ── Head Yaw Navigation ───────────────────────────────────────────────────
HEAD_YAW_LEFT_RATIO  = 0.84      # easier — small turn left triggers
HEAD_YAW_RIGHT_RATIO = 1.16      # easier — small turn right triggers
HEAD_YAW_HOLD_FRAMES = 12        # shorter hold needed
HEAD_YAW_COOLDOWN    = 1.2       # back to original

# ── Dwell: cursor near target → nearly freeze ─────────────────────────────
DWELL_RADIUS   = 6               # only locks when extremely still
DWELL_ALPHA    = 0.02            # slightly more responsive when dwelling

face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
screen_w, screen_h = pg.size()

heatmap = np.zeros((screen_h, screen_w), dtype=np.float32)
look_counter = {k: 0 for k in [
    "CENTER","TOP","BOTTOM","LEFT","RIGHT",
    "TOP_LEFT","TOP_RIGHT","BOTTOM_LEFT","BOTTOM_RIGHT"
]}

# ─────────────────────────────────────────────────────────────────────────
def euclidean(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_ear_6(landmarks, eye_side):
    if eye_side == "LEFT":
        p1,p2,p3 = landmarks[33],  landmarks[160], landmarks[158]
        p4,p5,p6 = landmarks[133], landmarks[153], landmarks[144]
    else:
        p1,p2,p3 = landmarks[362], landmarks[385], landmarks[387]
        p4,p5,p6 = landmarks[263], landmarks[373], landmarks[380]
    A = euclidean(p2, p6)
    B = euclidean(p3, p5)
    C = euclidean(p1, p4)
    return (A + B) / (2.0 * C + 1e-6)

def get_head_pitch_metric(landmarks):
    eye_span = euclidean(landmarks[33], landmarks[263])
    return (landmarks[152].y - landmarks[1].y) / max(eye_span, 1e-6)

def get_head_yaw_metric(landmarks):
    nose_x      = landmarks[1].x
    left_ear_x  = landmarks[234].x
    right_ear_x = landmarks[454].x
    span = right_ear_x - left_ear_x
    if abs(span) < 1e-4:
        return 0.5
    return float(np.clip((nose_x - left_ear_x) / span, 0.0, 1.0))

def draw_target(frame, label):
    h, w = frame.shape[:2]
    pos = {"CENTER":(w//2,h//2),"UP":(w//2,h//5),"DOWN":(w//2,(4*h)//5),
           "LEFT":(w//5,h//2),"RIGHT":((4*w)//5,h//2)}
    x, y = pos[label]
    cv2.circle(frame, (x,y), 18, (0,255,255), 3)
    cv2.circle(frame, (x,y),  5, (0,255,255), -1)

def process_face_landmarks(frame):
    small = cv2.resize(frame, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)
    return face_mesh.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

def apply_response_curve(value, exponent):
    value = float(np.clip(value, 0.0, 1.0))
    if value == 0.5:
        return 0.5
    offset = (value - 0.5) * 2.0
    curved = np.sign(offset) * (abs(offset) ** exponent)
    return float(np.clip((curved + 1.0) / 2.0, 0.0, 1.0))

def apply_deadzone(value, deadzone):
    offset = value - 0.5
    if abs(offset) <= deadzone:
        return 0.5
    scaled = (abs(offset) - deadzone) / max(0.5 - deadzone, 1e-6)
    return float(np.clip(0.5 + np.sign(offset) * scaled * 0.5, 0.0, 1.0))

def run_gaze_calibration(camera, face_mesh):
    print("\nGaze calibration — follow the dot with your eyes only.\n")
    samples = {name: [] for name, _ in CALIBRATION_POINTS}
    for name, instruction in CALIBRATION_POINTS:
        start = time.time()
        while time.time() - start < CALIBRATION_SECONDS:
            ret, frame = camera.read()
            if not ret:
                continue
            frame  = cv2.flip(frame, 1)
            output = process_face_landmarks(frame)
            if output.multi_face_landmarks:
                iris = output.multi_face_landmarks[0].landmark[468]
                samples[name].append((iris.x, iris.y))
            remaining = max(0.0, CALIBRATION_SECONDS - (time.time() - start))
            draw_target(frame, name)
            cv2.putText(frame, "Gaze calibration", (25,40),  1, 1.5, (0,255,0),    2)
            cv2.putText(frame, instruction,         (25,80),  1, 1.4, (255,255,255),2)
            cv2.putText(frame, f"{remaining:.1f}s", (25,118), 1, 1.3, (0,255,255),  2)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt
    cv2.destroyWindow("Calibration")

    defaults = {"CENTER":(0.50,0.50),"UP":(0.50,0.38),
                "DOWN":(0.50,0.62),"LEFT":(0.40,0.50),"RIGHT":(0.60,0.50)}
    calibrated = {}
    for name, _ in CALIBRATION_POINTS:
        pts = samples[name]
        calibrated[name] = (float(np.median([p[0] for p in pts])),
                             float(np.median([p[1] for p in pts]))) if pts else defaults[name]

    cx,cy = calibrated["CENTER"]
    lx,_  = calibrated["LEFT"]
    rx,_  = calibrated["RIGHT"]
    _,uy  = calibrated["UP"]
    _,dy  = calibrated["DOWN"]
    x_span= max(abs(rx-lx),0.08)
    y_span= max(abs(dy-uy),0.08)
    xm    = max(0.02, x_span*(1.0-CURSOR_RANGE_SCALE)*0.5)
    yhr   = max(0.04, (y_span*CURSOR_RANGE_SCALE)/2.0)
    swh   = max(0.04, x_span*0.18)
    config= {
        "gaze_x_min":       float(np.clip(min(lx,rx)-xm, 0,1)),
        "gaze_x_max":       float(np.clip(max(lx,rx)+xm, 0,1)),
        "gaze_y_min":       float(np.clip(cy-yhr,         0,1)),
        "gaze_y_max":       float(np.clip(cy+yhr,         0,1)),
        "scroll_zone_min_x":float(np.clip(cx-swh,         0,1)),
        "scroll_zone_max_x":float(np.clip(cx+swh,         0,1)),
        "scroll_top_y":     float(np.clip((cy+uy)/2,      0,1)),
        "scroll_bottom_y":  float(np.clip((cy+dy)/2,      0,1)),
    }
    print("Gaze calibration complete.")
    print(f" Cursor X: {config['gaze_x_min']:.3f} → {config['gaze_x_max']:.3f}")
    print(f" Cursor Y: {config['gaze_y_min']:.3f} → {config['gaze_y_max']:.3f}\n")
    return config

# ─────────────────────────────────────────────────────────────────────────
# INITIAL EAR CALIBRATION
# ─────────────────────────────────────────────────────────────────────────
print("=" * 55)
print(" EyeMouse — Dynamic Threshold Version")
print("=" * 55)
print("\nKeep BOTH EYES OPEN (3 sec)...")

calib_ear_left   = []
calib_ear_right  = []
calib_head_pitch = []
calib_head_yaw   = []

start = time.time()
while time.time() - start < 3:
    ret, frame = camera.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    res   = process_face_landmarks(frame)
    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0].landmark
        calib_ear_left.append(get_ear_6(lms, "LEFT"))
        calib_ear_right.append(get_ear_6(lms, "RIGHT"))
        calib_head_pitch.append(get_head_pitch_metric(lms))
        calib_head_yaw.append(get_head_yaw_metric(lms))
    remaining = int(3 - (time.time() - start)) + 1
    cv2.putText(frame, f"Calibrating... {remaining}s", (30,50), 1, 2, (0,255,0), 2)
    cv2.imshow('Calibration', frame)
    cv2.waitKey(1)
cv2.destroyWindow('Calibration')

left_baseline_init  = float(np.mean(calib_ear_left))  if calib_ear_left  else 0.25
right_baseline_init = float(np.mean(calib_ear_right)) if calib_ear_right else 0.25
head_pitch_baseline = float(np.mean(calib_head_pitch)) if calib_head_pitch else 1.0
head_yaw_baseline   = float(np.mean(calib_head_yaw))   if calib_head_yaw   else 0.5

left_baseline_live  = left_baseline_init
right_baseline_live = right_baseline_init

left_ear_window  = deque(maxlen=DYNAMIC_BASELINE_WINDOW)
right_ear_window = deque(maxlen=DYNAMIC_BASELINE_WINDOW)
for v in calib_ear_left[-DYNAMIC_BASELINE_WINDOW:]:
    left_ear_window.append(v)
for v in calib_ear_right[-DYNAMIC_BASELINE_WINDOW:]:
    right_ear_window.append(v)

HEAD_SCROLL_UP_THR   = head_pitch_baseline * HEAD_SCROLL_UP_RATIO
HEAD_SCROLL_DOWN_THR = head_pitch_baseline * HEAD_SCROLL_DOWN_RATIO
HEAD_YAW_LEFT_THR    = head_yaw_baseline   * HEAD_YAW_LEFT_RATIO
HEAD_YAW_RIGHT_THR   = head_yaw_baseline   * HEAD_YAW_RIGHT_RATIO

print(f"\n Left  baseline (init) : {left_baseline_init:.4f}")
print(f" Right baseline (init) : {right_baseline_init:.4f}")
print(f" WINK_RATIO            : {WINK_RATIO}  (thr = baseline x {WINK_RATIO})")
print(f" OPEN_RATIO            : {OPEN_RATIO}")
print(f" MIN_ASYMMETRY         : {MIN_ASYMMETRY}")
print(f" Head pitch neutral    : {head_pitch_baseline:.4f}")
print(f" Head yaw   neutral    : {head_yaw_baseline:.4f}")
print(f" Yaw left / right thr  : {HEAD_YAW_LEFT_THR:.4f} / {HEAD_YAW_RIGHT_THR:.4f}")
print(f"\n [DYNAMIC] EAR baselines update every frame (window={DYNAMIC_BASELINE_WINDOW})")
print()

gaze_config = run_gaze_calibration(camera, face_mesh)

# ─────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────
left_wink_active_frames  = 0
right_wink_active_frames = 0
scroll_up_count   = 0
scroll_down_count = 0
scroll_up_armed   = True
scroll_down_armed = True
yaw_left_count    = 0
yaw_right_count   = 0
last_yaw_time     = 0.0

last_left_click         = 0.0
last_right_click        = 0.0
last_scroll_time        = 0.0
last_left_wink_release  = 0.0
pending_left_click      = False
pending_left_click_time = 0.0
drag_mode               = False
last_eyes_open_time     = time.time()

smooth_x, smooth_y = screen_w // 2, screen_h // 2
frozen_click_x     = smooth_x
frozen_click_y     = smooth_y
wink_freeze_active = False

# Dwell state
prev_smooth_x = smooth_x
prev_smooth_y = smooth_y

system_on    = True
last_toggle  = 0.0
total_frames = 0

print("Running. Controls:")
print(" Left wink        → Left click (wink again quickly for double-click)")
print(" Right wink       → Right click  (cursor freezes on wink start)")
print(" Long right wink  → Drag / drop toggle")
print(" Chin up          → Scroll up")
print(" Chin down        → Scroll down")
print(" Head turn LEFT   → Alt+Left  (browser back)")
print(" Head turn RIGHT  → Alt+Right (browser forward)")
print(" Open mouth       → Toggle system on/off")
print(" Q → Quit  |  H → Heatmap\n")

# ─────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────
while True:
    loop_start = time.time()
    ret, camera_frame = camera.read()
    if not ret:
        break

    frame  = cv2.flip(camera_frame, 1)
    output = process_face_landmarks(frame)

    sc = (0,255,100) if system_on else (0,60,200)
    cv2.putText(frame, f"System: {'ON' if system_on else 'OFF'}",
                (frame.shape[1]-160, 35), 1, 1.8, sc, 2)

    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark

        # ── MOUTH TOGGLE
        mouth_dist = landmarks[14].y - landmarks[13].y
        if mouth_dist > MOUTH_THRESHOLD and (time.time() - last_toggle > 1.2):
            system_on   = not system_on
            last_toggle = time.time()
            print(f"System {'ENABLED' if system_on else 'DISABLED'}")

        if system_on:
            total_frames += 1

            left_ear  = get_ear_6(landmarks, "LEFT")
            right_ear = get_ear_6(landmarks, "RIGHT")

            # ── DYNAMIC BASELINE UPDATE
            if left_ear > left_baseline_live * DYNAMIC_UPDATE_GUARD:
                left_ear_window.append(left_ear)
            if right_ear > right_baseline_live * DYNAMIC_UPDATE_GUARD:
                right_ear_window.append(right_ear)

            if len(left_ear_window) >= 10:
                lm = float(np.median(left_ear_window))
                if lm >= left_baseline_live * 0.98:
                    left_baseline_live = (DYNAMIC_ALPHA * lm
                                          + (1 - DYNAMIC_ALPHA) * left_baseline_live)

            if len(right_ear_window) >= 10:
                rm = float(np.median(right_ear_window))
                if rm >= right_baseline_live * 0.98:
                    right_baseline_live = (DYNAMIC_ALPHA * rm
                                           + (1 - DYNAMIC_ALPHA) * right_baseline_live)

            L_WINK_THR       = left_baseline_live  * WINK_RATIO
            R_WINK_THR       = right_baseline_live * WINK_RATIO
            L_OPEN_THR       = left_baseline_live  * OPEN_RATIO
            R_OPEN_THR       = right_baseline_live * OPEN_RATIO
            L_WINK_START_THR = left_baseline_live  * WINK_START_RATIO
            R_WINK_START_THR = right_baseline_live * WINK_START_RATIO

            asymmetry = abs(left_ear - right_ear)

            left_winking = (
                left_ear  < L_WINK_THR and
                right_ear > R_OPEN_THR and
                asymmetry > MIN_ASYMMETRY
            )
            right_winking = (
                right_ear < R_WINK_THR and
                left_ear  > L_OPEN_THR and
                asymmetry > MIN_ASYMMETRY
            )

            now          = time.time()

            # ── CURSOR FREEZE
            wink_building = left_winking or right_winking
            if wink_building:
                if not wink_freeze_active:
                    frozen_click_x = smooth_x
                    frozen_click_y = smooth_y
                    wink_freeze_active = True
            else:
                # Only clear freeze if no pending click AND no active wink frames
                if (left_wink_active_frames == 0 and right_wink_active_frames == 0
                        and not pending_left_click
                        and (now - last_left_click)  > CLICK_MOTION_BLOCK
                        and (now - last_right_click) > CLICK_MOTION_BLOCK):
                    wink_freeze_active = False

            prev_left_frames  = left_wink_active_frames
            prev_right_frames = right_wink_active_frames
            left_wink_active_frames  = left_wink_active_frames  + 1 if left_winking  else 0
            right_wink_active_frames = right_wink_active_frames + 1 if right_winking else 0

            is_clicking  = False
            is_scrolling = False
            status_text  = None
            status_color = (255, 255, 255)

            # Pending single left click confirmation
            if pending_left_click and (now - pending_left_click_time) > DOUBLE_WINK_GAP:
                if (now - last_left_click) > LEFT_COOLDOWN:
                    # Move to frozen position first, then click — captures button precisely
                    cx_l = int(np.clip(frozen_click_x, 20, screen_w-20))
                    cy_l = int(np.clip(frozen_click_y, 20, screen_h-20))
                    pg.moveTo(cx_l, cy_l)
                    pg.click(button='left')
                    last_left_click = now
                    is_clicking     = True
                    status_text     = "LEFT CLICK"
                    status_color    = (0, 60, 255)
                    print(f" LEFT CLICK at ({cx_l},{cy_l}) | L:{left_ear:.3f}(thr {L_WINK_THR:.3f})"
                          f" R:{right_ear:.3f}(base {right_baseline_live:.3f})")
                pending_left_click = False

            if prev_left_frames >= WINK_CONFIRM_FRAMES and not left_winking:
                if pending_left_click and (now - pending_left_click_time) <= DOUBLE_WINK_GAP:
                    if (now - last_left_click) > LEFT_COOLDOWN:
                        cx_l = int(np.clip(frozen_click_x, 20, screen_w-20))
                        cy_l = int(np.clip(frozen_click_y, 20, screen_h-20))
                        pg.moveTo(cx_l, cy_l)
                        pg.doubleClick()
                        last_left_click = now
                        is_clicking     = True
                        status_text     = "DOUBLE CLICK"
                        status_color    = (255, 120, 0)
                        print(f" DOUBLE CLICK at ({cx_l},{cy_l}) | L:{left_ear:.3f} R:{right_ear:.3f}")
                    pending_left_click = False
                else:
                    pending_left_click      = True
                    pending_left_click_time = now
                    last_left_wink_release  = now

            if prev_right_frames >= DRAG_HOLD_FRAMES and not right_winking:
                if drag_mode:
                    pg.mouseUp()
                    drag_mode   = False
                    status_text = "DROP"
                    status_color= (0,220,255)
                    print(" DROP")
                else:
                    pg.mouseDown()
                    drag_mode   = True
                    status_text = "DRAG ON"
                    status_color= (0,220,255)
                    print(" DRAG START")
                is_clicking = True
            elif prev_right_frames >= WINK_CONFIRM_FRAMES and not right_winking:
                if (now - last_right_click) > RIGHT_COOLDOWN:
                    cx_c = int(np.clip(frozen_click_x, 20, screen_w-20))
                    cy_c = int(np.clip(frozen_click_y, 20, screen_h-20))
                    pg.moveTo(cx_c, cy_c)
                    pg.click(button='right')
                    last_right_click = now
                    is_clicking      = True
                    status_text      = "RIGHT CLICK"
                    status_color     = (0,165,255)
                    print(f" RIGHT CLICK at ({cx_c},{cy_c})"
                          f" | L:{left_ear:.3f} R:{right_ear:.3f}(thr {R_WINK_THR:.3f})")

            if status_text:
                cv2.putText(frame, status_text, (50,80), 1, 2.0, status_color, 3)

            iris             = landmarks[468]
            eyes_open        = left_ear > L_OPEN_THR and right_ear > R_OPEN_THR
            both_eyes_closed = left_ear < L_WINK_THR and right_ear < R_WINK_THR

            wink_starting = (
                left_winking or right_winking or
                left_wink_active_frames  > 0 or
                right_wink_active_frames > 0
            )
            if not both_eyes_closed:
                last_eyes_open_time = now

            allow_cursor_motion = (
                (now - last_eyes_open_time) <= EYE_OPEN_GRACE and
                not wink_starting and
                not both_eyes_closed and
                not wink_freeze_active and
                not is_clicking and
                (now - last_left_click)  > CLICK_MOTION_BLOCK and
                (now - last_right_click) > CLICK_MOTION_BLOCK
            )

            head_pitch = get_head_pitch_metric(landmarks)
            head_yaw   = get_head_yaw_metric(landmarks)

            scroll_ready = (
                eyes_open and
                not left_winking and not right_winking and
                left_wink_active_frames  == 0 and
                right_wink_active_frames == 0
            )

            # ── SCROLL
            if   scroll_ready and head_pitch <= HEAD_SCROLL_UP_THR:
                scroll_up_count += 1; scroll_down_count = 0; scroll_down_armed = True
            elif scroll_ready and head_pitch >= HEAD_SCROLL_DOWN_THR:
                scroll_down_count += 1; scroll_up_count = 0; scroll_up_armed = True
            else:
                scroll_up_count = scroll_down_count = 0
                scroll_up_armed = scroll_down_armed = True

            if not is_clicking and (now - last_scroll_time) > SCROLL_COOLDOWN:
                if scroll_up_armed and scroll_up_count >= SCROLL_HOLD_FRAMES:
                    pg.scroll(SCROLL_AMOUNT)
                    last_scroll_time = now; is_scrolling = True
                    scroll_up_count = 0; scroll_up_armed = False
                    print(" SCROLL UP")
                    cv2.putText(frame,"SCROLL UP",(50,80),1,2.0,(80,255,80),3)
                elif scroll_down_armed and scroll_down_count >= SCROLL_HOLD_FRAMES:
                    pg.scroll(-SCROLL_AMOUNT)
                    last_scroll_time = now; is_scrolling = True
                    scroll_down_count = 0; scroll_down_armed = False
                    print(" SCROLL DOWN")
                    cv2.putText(frame,"SCROLL DOWN",(50,80),1,2.0,(80,255,80),3)

            # ── HEAD YAW NAV
            yaw_ready = (
                scroll_ready and not is_scrolling and not is_clicking and
                (now - last_yaw_time) > HEAD_YAW_COOLDOWN
            )
            if yaw_ready:
                if   head_yaw <= HEAD_YAW_LEFT_THR:
                    yaw_left_count += 1; yaw_right_count = 0
                elif head_yaw >= HEAD_YAW_RIGHT_THR:
                    yaw_right_count += 1; yaw_left_count = 0
                else:
                    yaw_left_count = yaw_right_count = 0

                if yaw_left_count >= HEAD_YAW_HOLD_FRAMES:
                    pg.hotkey('alt','left')
                    last_yaw_time = now; yaw_left_count = 0
                    print(f" HEAD LEFT → Alt+Left (yaw={head_yaw:.3f})")
                    cv2.putText(frame,"HEAD LEFT",(50,80),1,2.0,(255,200,0),3)
                elif yaw_right_count >= HEAD_YAW_HOLD_FRAMES:
                    pg.hotkey('alt','right')
                    last_yaw_time = now; yaw_right_count = 0
                    print(f" HEAD RIGHT → Alt+Right (yaw={head_yaw:.3f})")
                    cv2.putText(frame,"HEAD RIGHT",(50,80),1,2.0,(255,200,0),3)
            else:
                yaw_left_count = yaw_right_count = 0

            # ── CURSOR MOVEMENT
            if not is_clicking and not is_scrolling and allow_cursor_motion:
                xr     = max(gaze_config["gaze_x_max"]-gaze_config["gaze_x_min"],1e-6)
                yr     = max(gaze_config["gaze_y_max"]-gaze_config["gaze_y_min"],1e-6)
                norm_x = (iris.x - gaze_config["gaze_x_min"]) / xr
                norm_y = (iris.y - gaze_config["gaze_y_min"]) / yr

                # Direct target — where eye is looking right now
                tx = float(np.clip(norm_x, 0.0, 1.0)) * screen_w
                ty = float(np.clip(norm_y, 0.0, 1.0)) * screen_h

                # Distance from current smooth position to target
                dist = np.hypot(tx - smooth_x, ty - smooth_y)

                # Speed scales with distance — far = fast, near = slow
                if dist > 300:
                    blend = 0.14          # fast — big eye movement
                elif dist > 150:
                    blend = 0.08          # medium
                elif dist > 80:
                    blend = 0.04          # slowing down near area
                elif dist > 40:
                    blend = 0.018         # very slow — close to button
                elif dist > 15:
                    blend = 0.008         # crawling — right on button
                else:
                    blend = 0.004         # frozen — locked on

                smooth_x = smooth_x + (tx - smooth_x) * blend
                smooth_y = smooth_y + (ty - smooth_y) * blend

                cx = int(np.clip(smooth_x, 20, screen_w-20))
                cy = int(np.clip(smooth_y, 20, screen_h-20))
                pg.moveTo(cx, cy)
                cv2.circle(heatmap, (cx,cy), 20, 1, -1)

                prev_smooth_x = smooth_x
                prev_smooth_y = smooth_y

            # ── ROI
            x_z  = ("LEFT" if iris.x<0.45 else "RIGHT" if iris.x>0.55 else "CENTER")
            y_z  = ("TOP"  if iris.y<0.45 else "BOTTOM" if iris.y>0.55 else "CENTER")
            gaze = (f"{y_z}_{x_z}" if x_z!="CENTER" and y_z!="CENTER"
                    else x_z if x_z!="CENTER" else y_z)
            look_counter[gaze] += 1

            # ── HUD
            cv2.putText(frame, f"ROI: {gaze}", (50,115), 1, 1.8, (255,255,0), 2)
            cv2.putText(frame, f"Drag: {'ON' if drag_mode else 'OFF'}", (50,340), 1, 1.3,
                        (0,220,255) if drag_mode else (200,200,200), 2)
            cv2.putText(frame, f"Track: {'OK' if allow_cursor_motion else 'PAUSED'}", (50,365),
                        1, 1.1, (80,255,80) if allow_cursor_motion else (0,170,255), 2)
            cv2.putText(frame, f"Pitch:{head_pitch:.3f}  Yaw:{head_yaw:.3f}",
                        (50,390), 1, 1.1, (200,220,255), 2)

            h_f = frame.shape[0]
            cv2.putText(frame,
                f"L:{left_ear:.3f} base:{left_baseline_live:.3f} thr:{L_WINK_THR:.3f}",
                (50, h_f-55), 1, 1.2, (180,255,180), 2)
            cv2.putText(frame,
                f"R:{right_ear:.3f} base:{right_baseline_live:.3f} thr:{R_WINK_THR:.3f}",
                (50, h_f-30), 1, 1.2, (180,255,180), 2)

            for frames, color, label, y0 in [
                (left_wink_active_frames,  (0, 80,255),  "L-WINK", 140),
                (right_wink_active_frames, (0,165,255),  "R-WINK", 165),
            ]:
                if frames > 0:
                    bw = int((frames/DRAG_HOLD_FRAMES)*150)
                    cv2.rectangle(frame,(50,y0),(50+bw,y0+18),color,-1)
                    cv2.putText(frame,label,(205,y0+15),1,1.3,color,2)

            for count, color, label, y0, mx in [
                (scroll_up_count,   (80,255,80),  "SCROLL UP",   190, SCROLL_HOLD_FRAMES),
                (scroll_down_count, (80,220,120), "SCROLL DOWN", 215, SCROLL_HOLD_FRAMES),
                (yaw_left_count,    (255,200,0),  "HEAD LEFT",   240, HEAD_YAW_HOLD_FRAMES),
                (yaw_right_count,   (255,200,0),  "HEAD RIGHT",  265, HEAD_YAW_HOLD_FRAMES),
            ]:
                if count > 0:
                    bw = int((count/mx)*150)
                    cv2.rectangle(frame,(50,y0),(50+min(bw,150),y0+18),color,-1)
                    cv2.putText(frame,label,(205,y0+15),1,1.2,color,2)

    small_frame = cv2.resize(frame, (320, 240))
    cv2.imshow('EyeMouse', small_frame)
    cv2.moveWindow('EyeMouse', 20, 20)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        hb = cv2.GaussianBlur(heatmap,(31,31),0)
        hn = cv2.normalize(hb,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("Gaze Heatmap", cv2.applyColorMap(hn, cv2.COLORMAP_JET))
    elif key == ord('x') and drag_mode:
        pg.mouseUp(); drag_mode = False; print(" DRAG CANCELLED")

    elapsed = time.time() - loop_start
    delay   = max(0.0, (1.0/TARGET_FPS) - elapsed)
    if delay > 0:
        time.sleep(delay)

camera.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(" FINAL GAZE ROI ANALYSIS")
print("="*55)
most_looked_roi, max_frames = "None", 0
for d in ["TOP_LEFT","TOP","TOP_RIGHT","LEFT","CENTER","RIGHT",
          "BOTTOM_LEFT","BOTTOM","BOTTOM_RIGHT"]:
    f = look_counter[d]
    p = (f/total_frames*100) if total_frames > 0 else 0
    print(f" {d:12} | {'█'*int(p/2):<50} {p:5.1f}%")
    if f > max_frames:
        max_frames = f; most_looked_roi = d
print("-"*55)
print(f" PRIMARY ROI     : {most_looked_roi}")
print(f" Active Frames   : {total_frames}")
print(f" Final L-baseline: {left_baseline_live:.4f}  (init: {left_baseline_init:.4f})")
print(f" Final R-baseline: {right_baseline_live:.4f}  (init: {right_baseline_init:.4f})")
print("="*55)