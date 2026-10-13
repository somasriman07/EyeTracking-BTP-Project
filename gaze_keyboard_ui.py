"""
gaze_keyboard_ui.py
-------------------
Interactive Virtual On-Screen Gaze Keyboard Renderer with Dwell Time Feedback.
Designed for Assistive Human-Computer Interaction (HCI).
Supports QWERTY layout, dynamic word predictions, quick phrases, visual dwell filling,
and non-blocking Text-to-Speech (TTS).
"""

import cv2
import numpy as np
import time
import threading
import pyttsx3
from word_predictor import WordPredictor

# Initialize TTS in background thread worker
_tts_lock = threading.Lock()
def speak_text_async(text):
    if not text.strip():
        return
    def _worker():
        with _tts_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS Error]: {e}")
    threading.Thread(target=_worker, daemon=True).start()


class KeyButton:
    def __init__(self, key_id, label, x, y, w, h, key_type="char"):
        self.key_id = key_id
        self.label = label
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.key_type = key_type  # "char", "action", "prediction", "phrase"

    def contains(self, px, py):
        return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)


class GazeKeyboard:
    def __init__(self, screen_w=1280, screen_h=720, dwell_time=0.75):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.dwell_time = dwell_time  # Seconds to trigger dwell click

        self.predictor = WordPredictor()
        self.text = ""
        self.last_word_prefix = ""
        self.predictions = self.predictor.predict("")

        # Dwell state tracking
        self.active_key = None
        self.hover_start_time = 0.0
        self.last_trigger_time = 0.0
        self.trigger_cooldown = 0.35  # Prevent accidental rapid double triggers
        self.last_triggered_key_id = None
        self.flash_timer = 0.0

        # UI Layout setup
        self.keys = []
        self._build_layout()

    def _build_layout(self):
        """Builds an eye-tracking friendly, large-button layout."""
        self.keys.clear()
        sw, sh = self.screen_w, self.screen_h

        # Keyboard area occupies bottom 65% of screen
        kb_top = int(sh * 0.32)
        kb_h = sh - kb_top - 20
        margin_x = 40

        # 1. Prediction Bar (4 big chips)
        pred_y = kb_top
        pred_h = int(kb_h * 0.14)
        pred_w = (sw - 2 * margin_x - 30) // 4
        for i in range(4):
            kx = margin_x + i * (pred_w + 10)
            self.keys.append(KeyButton(f"PRED_{i}", f"word{i+1}", kx, pred_y, pred_w, pred_h, "prediction"))

        # 2. QWERTY Rows
        row_y_start = pred_y + pred_h + 12
        row_h = int(kb_h * 0.16)
        row_gap = 10

        # Row 1: Q W E R T Y U I O P
        row1 = list("QWERTYUIOP")
        btn_w_r1 = (sw - 2 * margin_x - (len(row1) - 1) * 8) // len(row1)
        r1_y = row_y_start
        for i, char in enumerate(row1):
            kx = margin_x + i * (btn_w_r1 + 8)
            self.keys.append(KeyButton(char, char, kx, r1_y, btn_w_r1, row_h, "char"))

        # Row 2: A S D F G H J K L
        row2 = list("ASDFGHJKL")
        btn_w_r2 = (sw - 2 * margin_x - 60 - (len(row2) - 1) * 8) // len(row2)
        r2_y = r1_y + row_h + row_gap
        r2_offset = margin_x + 30
        for i, char in enumerate(row2):
            kx = r2_offset + i * (btn_w_r2 + 8)
            self.keys.append(KeyButton(char, char, kx, r2_y, btn_w_r2, row_h, "char"))

        # Row 3: Z X C V B N M
        row3 = list("ZXCVBNM")
        btn_w_r3 = (sw - 2 * margin_x - 120 - (len(row3) - 1) * 8) // len(row3)
        r3_y = r2_y + row_h + row_gap
        r3_offset = margin_x + 60
        for i, char in enumerate(row3):
            kx = r3_offset + i * (btn_w_r3 + 8)
            self.keys.append(KeyButton(char, char, kx, r3_y, btn_w_r3, row_h, "char"))

        # Row 4: Action Keys [SPACE], [BACKSPACE], [SPEAK], [CLEAR], [MODE/CLOSE]
        r4_y = r3_y + row_h + row_gap
        action_keys = [
            ("SPACE", "SPACE", int(sw * 0.28)),
            ("BACK", "<- BKSP", int(sw * 0.16)),
            ("SPEAK", "SPEAK (TTS)", int(sw * 0.18)),
            ("CLEAR", "CLEAR", int(sw * 0.13)),
            ("TOGGLE_MOUSE", "EXIT TO MOUSE", int(sw * 0.17)),
        ]
        curr_x = margin_x
        for kid, lbl, kw in action_keys:
            self.keys.append(KeyButton(kid, lbl, curr_x, r4_y, kw, row_h, "action"))
            curr_x += kw + 10

        self._refresh_prediction_labels()

    def _refresh_prediction_labels(self):
        """Updates the prediction key labels based on current typed word prefix."""
        words = self.text.split()
        if self.text.endswith(" ") or not words:
            self.last_word_prefix = ""
        else:
            self.last_word_prefix = words[-1]

        self.predictions = self.predictor.predict(self.last_word_prefix, max_results=4)
        pred_idx = 0
        for k in self.keys:
            if k.key_type == "prediction":
                if pred_idx < len(self.predictions):
                    k.label = self.predictions[pred_idx]
                else:
                    k.label = ""
                pred_idx += 1

    def update_gaze(self, gaze_x, gaze_y, wink_confirm=False):
        """
        Processes gaze hover position and optional instant wink confirmation.
        Returns: (event_triggered, key_label)
        """
        now = time.time()
        hovered_key = None

        if gaze_x is not None and gaze_y is not None:
            for k in self.keys:
                if k.contains(gaze_x, gaze_y):
                    hovered_key = k
                    break

        triggered = False
        triggered_label = None

        # Cooldown check
        if now - self.last_trigger_time < self.trigger_cooldown:
            return False, None

        if hovered_key is not None:
            if wink_confirm:
                # Instant trigger on wink
                triggered = True
                triggered_label = hovered_key.label
                self._execute_key(hovered_key)
                self.last_trigger_time = now
                self.last_triggered_key_id = hovered_key.key_id
                self.flash_timer = now + 0.20
                self.active_key = hovered_key
                self.hover_start_time = now + 999.0
            elif hovered_key != self.active_key:
                # Started hovering a new key
                self.active_key = hovered_key
                self.hover_start_time = now
            else:
                # Hovering the same key: check dwell completion
                dwell_elapsed = now - self.hover_start_time
                if dwell_elapsed >= self.dwell_time:
                    triggered = True
                    triggered_label = hovered_key.label
                    self._execute_key(hovered_key)
                    self.last_trigger_time = now
                    self.last_triggered_key_id = hovered_key.key_id
                    self.flash_timer = now + 0.20
                    self.hover_start_time = now + 999.0  # Reset hover
        else:
            self.active_key = None
            self.hover_start_time = 0.0

        return triggered, triggered_label

    def _execute_key(self, key):
        """Applies the key logic to the current text buffer."""
        kid = key.key_id

        if key.key_type == "char":
            self.text += key.label
        elif kid == "SPACE":
            self.text += " "
        elif kid == "BACK":
            if len(self.text) > 0:
                self.text = self.text[:-1]
        elif kid == "CLEAR":
            self.text = ""
        elif kid == "SPEAK":
            speak_text_async(self.text)
        elif kid == "TOGGLE_MOUSE":
            # Handled by main caller
            pass
        elif key.key_type == "prediction" and key.label:
            # Replace current prefix with completed prediction
            words = self.text.rstrip().split()
            if self.last_word_prefix and words:
                words[-1] = key.label
                self.text = " ".join(words) + " "
            else:
                self.text += key.label + " "

        self._refresh_prediction_labels()

    def render(self, frame_or_canvas, gaze_pt=None):
        """
        Renders the virtual keyboard, text box, predictions, and dwell indicator.
        """
        now = time.time()
        canvas = frame_or_canvas
        sh, sw = canvas.shape[:2]

        # 1. Background overlay panel
        overlay = canvas.copy()
        cv2.rectangle(overlay, (20, 10), (sw - 20, sh - 15), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)

        # 2. Header Title & Instructions
        cv2.putText(canvas, "ASSISTIVE EYE-GAZE KEYBOARD", (40, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 230, 255), 2, cv2.LINE_AA)
        status_info = f"Dwell Threshold: {self.dwell_time:.2f}s  |  Hold Gaze or Wink Left Eye to Type"
        cv2.putText(canvas, status_info, (sw - 680, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 195, 210), 1, cv2.LINE_AA)

        # 3. Text Display Box
        tb_x, tb_y, tb_w, tb_h = 40, 58, sw - 80, int(sh * 0.16)
        cv2.rectangle(canvas, (tb_x, tb_y), (tb_x + tb_w, tb_y + tb_h), (28, 32, 42), -1)
        cv2.rectangle(canvas, (tb_x, tb_y), (tb_x + tb_w, tb_y + tb_h), (80, 95, 120), 2)

        # Composed text with blinking cursor
        cursor_char = "|" if (int(now * 2) % 2 == 0) else ""
        display_str = self.text + cursor_char
        # Wrap or truncate to fit display box
        if len(display_str) > 55:
            display_str = "..." + display_str[-52:]

        cv2.putText(canvas, display_str, (tb_x + 20, tb_y + tb_h // 2 + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.15, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Draw All Keys
        dwell_pct = 0.0
        if self.active_key is not None:
            dwell_pct = min(1.0, max(0.0, (now - self.hover_start_time) / self.dwell_time))

        for k in self.keys:
            is_hovered = (k == self.active_key)
            is_flashing = (k.key_id == self.last_triggered_key_id and now < self.flash_timer)

            # Color styling based on key type and hover state
            if is_flashing:
                bg_color = (0, 255, 100)      # Bright Flash Green
                border_color = (255, 255, 255)
                text_color = (0, 0, 0)
            elif is_hovered:
                bg_color = (45, 65, 85)        # Hover slate blue
                border_color = (0, 230, 255)   # Cyan border
                text_color = (0, 255, 255)
            elif k.key_type == "prediction":
                bg_color = (35, 45, 60)        # Prediction chip color
                border_color = (65, 90, 120)
                text_color = (180, 220, 255)
            elif k.key_type == "action":
                if k.key_id == "SPEAK":
                    bg_color = (25, 70, 35)    # Emerald for speech
                    border_color = (50, 180, 80)
                    text_color = (180, 255, 180)
                elif k.key_id in ("CLEAR", "BACK"):
                    bg_color = (65, 30, 30)    # Soft red for delete
                    border_color = (160, 60, 60)
                    text_color = (255, 190, 190)
                elif k.key_id == "TOGGLE_MOUSE":
                    bg_color = (60, 45, 20)    # Amber for mode switch
                    border_color = (180, 130, 40)
                    text_color = (255, 210, 130)
                else:
                    bg_color = (32, 38, 50)
                    border_color = (70, 85, 105)
                    text_color = (230, 230, 230)
            else:
                bg_color = (28, 32, 42)        # Standard dark key
                border_color = (55, 65, 85)
                text_color = (240, 240, 240)

            # Draw key background & border
            cv2.rectangle(canvas, (k.x, k.y), (k.x + k.w, k.y + k.h), bg_color, -1)

            # If hovered: draw animated dwell loading fill
            if is_hovered and dwell_pct > 0.0:
                fill_w = int(k.w * dwell_pct)
                cv2.rectangle(canvas, (k.x, k.y + k.h - 8), (k.x + fill_w, k.y + k.h),
                              (0, 255, 230), -1)

            cv2.rectangle(canvas, (k.x, k.y), (k.x + k.w, k.y + k.h), border_color, 2)

            # Draw key text centered
            font_scale = 0.85 if len(k.label) <= 2 else 0.60
            thickness = 2 if len(k.label) <= 2 else 1
            (tw, th), _ = cv2.getTextSize(k.label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
            tx = k.x + (k.w - tw) // 2
            ty = k.y + (k.h + th) // 2
            cv2.putText(canvas, k.label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

        # 5. Draw live gaze fixation reticle
        if gaze_pt is not None and gaze_pt[0] is not None and gaze_pt[1] is not None:
            gx, gy = int(gaze_pt[0]), int(gaze_pt[1])
            # Draw outer ring
            ring_radius = 22
            cv2.circle(canvas, (gx, gy), ring_radius, (0, 200, 255), 2, cv2.LINE_AA)
            # If dwelling, draw filling circular arc
            if self.active_key is not None and dwell_pct > 0.0:
                end_angle = int(360 * dwell_pct)
                cv2.ellipse(canvas, (gx, gy), (ring_radius, ring_radius), -90, 0, end_angle,
                            (0, 255, 100), 3, cv2.LINE_AA)
            cv2.circle(canvas, (gx, gy), 4, (0, 255, 255), -1, cv2.LINE_AA)

        return canvas
