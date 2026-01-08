import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np
from collections import deque
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

# ================= AUDIO SETUP =================
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
min_vol, max_vol, _ = volume.GetVolumeRange()

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(2)

# ================= TIMING & STABILITY =================
ACTION_DELAY = 1.5
LOCK_TIME = 3.0
STABILITY_THRESHOLD = 15  # pixels for unlock
UPDATE_THRESHOLD = 1  # minimal % change to update volume/brightness

last_action_time = 0
prev_gesture = None

# buffers for smoothing
vol_buffer = deque(maxlen=7)
bri_buffer = deque(maxlen=7)

# unlock state
volume_unlocked = False
brightness_unlocked = False

stable_start_vol = None
stable_start_bri = None

prev_vol_dist = None
prev_bri_dist = None

last_vol_percent = None
last_bri_percent = None

# ================= UTIL FUNCTIONS =================
def count_fingers(hand):
    fingers = 0
    # Thumb (right hand)
    if hand.landmark[4].x < hand.landmark[3].x:
        fingers += 1
    tips = [8, 12, 16, 20]  # index, middle, ring, pinky
    for tip in tips:
        if hand.landmark[tip].y < hand.landmark[tip - 2].y:
            fingers += 1
    return fingers

def smooth(buffer, value):
    buffer.append(value)
    return sum(buffer) / len(buffer)

# ================= MAIN LOOP =================
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    current_time = time.time()

    gesture_text = "No Hand"

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            fingers = count_fingers(hand)

            # ================= PLAY / PAUSE / PREVIOUS =================
            if fingers != prev_gesture and current_time - last_action_time > ACTION_DELAY:
                if fingers in [0, 5]:
                    pyautogui.press("playpause")
                    gesture_text = "PLAY / PAUSE"
                    last_action_time = current_time
                elif fingers == 3:
                    pyautogui.press("prevtrack")
                    gesture_text = "PREVIOUS"
                    last_action_time = current_time
                prev_gesture = fingers

            # ================= VOLUME CONTROL =================
            if fingers == 2:
                x1, y1 = int(hand.landmark[4].x * w), int(hand.landmark[4].y * h)
                x2, y2 = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)
                dist = math.hypot(x2 - x1, y2 - y1)
                dist = smooth(vol_buffer, dist)
                vol_percent = int(np.interp(dist, [40, 200], [0, 100]))

                if not volume_unlocked:
                    # Unlock check
                    if stable_start_vol is None:
                        stable_start_vol = time.time()
                        prev_vol_dist = dist
                    elif abs(dist - prev_vol_dist) < STABILITY_THRESHOLD:
                        if time.time() - stable_start_vol >= LOCK_TIME:
                            volume_unlocked = True
                            last_vol_percent = vol_percent
                    else:
                        stable_start_vol = time.time()
                    prev_vol_dist = dist
                    gesture_text = "✌ Hold to unlock VOLUME"
                else:
                    # Continuous adjustment with threshold
                    if last_vol_percent is None or abs(vol_percent - last_vol_percent) >= UPDATE_THRESHOLD:
                        vol = np.interp(dist, [40, 200], [min_vol, max_vol])
                        volume.SetMasterVolumeLevel(vol, None)
                        last_vol_percent = vol_percent
                    gesture_text = f"🔊 VOLUME: {last_vol_percent}%"
            else:
                volume_unlocked = False
                stable_start_vol = None
                prev_vol_dist = None
                last_vol_percent = None
                vol_buffer.clear()

            # ================= BRIGHTNESS CONTROL =================
            if fingers == 4:
                x1, y1 = int(hand.landmark[4].x * w), int(hand.landmark[4].y * h)
                x2, y2 = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)
                dist = math.hypot(x2 - x1, y2 - y1)
                dist = smooth(bri_buffer, dist)
                bri_percent = int(np.interp(dist, [40, 200], [0, 100]))

                if not brightness_unlocked:
                    if stable_start_bri is None:
                        stable_start_bri = time.time()
                        prev_bri_dist = dist
                    elif abs(dist - prev_bri_dist) < STABILITY_THRESHOLD:
                        if time.time() - stable_start_bri >= LOCK_TIME:
                            brightness_unlocked = True
                            last_bri_percent = bri_percent
                    else:
                        stable_start_bri = time.time()
                    prev_bri_dist = dist
                    gesture_text = "🖐 Hold to unlock BRIGHTNESS"
                else:
                    if last_bri_percent is None or abs(bri_percent - last_bri_percent) >= UPDATE_THRESHOLD:
                        sbc.set_brightness(bri_percent)
                        last_bri_percent = bri_percent
                    gesture_text = f"🔆 BRIGHTNESS: {last_bri_percent}%"
            else:
                brightness_unlocked = False
                stable_start_bri = None
                prev_bri_dist = None
                last_bri_percent = None
                bri_buffer.clear()

    cv2.putText(frame, gesture_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Gesture Controlled Media Player", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
