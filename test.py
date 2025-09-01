#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pygame
import time
from gtts import gTTS
from serial.tools import list_ports
import joblib
import json
import math

# ====== ML Model imports ======
from sklearn.ensemble import RandomForestClassifier

# ====== Dobot (pydobot) ======
try:
    from pydobot import Dobot
except Exception:
    Dobot = None

# ------- CONFIG -------
SOUNDS_DIR = "sounds"
MODEL_DIR = "trained_models"
MODEL_FILE = os.path.join(MODEL_DIR, "asl_classifier.pkl")
MODEL_INFO_FILE = os.path.join(MODEL_DIR, "model_info.json")

os.makedirs(SOUNDS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Dobot settings
DOBOT_ENABLED = True
DOBOT_PORT = "/dev/ttyUSB0"
SAFE_Z = 50.0
RETURN_HOME = True
HOME_POSE = (155.15, 58.87, 37.87, 20.78)

# ML prediction settings
PREDICTION_BUFFER_SIZE = 25
MIN_CONFIDENCE = 0.5
HOLD_FRAMES = 8  
TRIGGER_COOLDOWN = 2.0

# ------- Mediapipe -------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
HL = mp_hands.HandLandmark

# ------- Audio -------
supported_chars = [chr(c) for c in range(ord('A'), ord('Z'))]
sound_cache = {}

pygame.init()
try:
    pygame.mixer.init()
except Exception as e:
    print("Warning: pygame.mixer.init() failed:", e)

def ensure_sound_for_char(char):
    if char in sound_cache and sound_cache[char] is not None:
        return sound_cache[char]
    filename = os.path.join(SOUNDS_DIR, f"{char}.mp3")
    if not os.path.exists(filename):
        try:
            tts = gTTS(text=char, lang="en")
            tts.save(filename)
            print(f"gTTS saved {filename}")
        except Exception as e:
            print(f"gTTS failed for {char}: {e}")
            return None
    try:
        sound = pygame.mixer.Sound(filename)
        sound_cache[char] = sound
        return sound
    except Exception as e:
        print(f"Failed to load sound {filename}: {e}")
        sound_cache[char] = None
        return None

for ch in supported_chars:
    ensure_sound_for_char(ch)

# ------- 4D positions -------
positions_4d = {
    "home": (155.15, 58.87, 37.87, 20.78),
    "block_1": (140.97, 126.50, -37.63, 41.90),
    "block_2": (188.56, 14.84, -36.73, 4.50),
    "drop_off_1": (192.14, 147.91, -37.70, 7.59),
    "drop_off_2": (237.50, 38.63, -38.24, 9.24),
}

# ------- Dobot control -------
dobot = None
dobot_busy = False
last_job_time = 0.0

def find_first_port():
    ports = list_ports.comports()
    return ports[0].device if ports else None

def _dobot_configure_motion(d):
    try:
        if hasattr(d, "clear_alarm_state"):
            d.clear_alarm_state()
    except Exception:
        pass

    try:
        if hasattr(d, "set_ptp_joint_params"):
            d.set_ptp_joint_params(200, 200, 200, 200, 200, 200, 200, 200)
        if hasattr(d, "set_ptp_common_params"):
            d.set_ptp_common_params(200, 200)
        if hasattr(d, "set_ptp_coordinate_params"):
            d.set_ptp_coordinate_params(200, 200, 200, 200)
    except Exception as e:
        print("[DOBOT] set PTP params failed:", e)

    try:
        if hasattr(d, "queued_cmd_stop"):
            d.queued_cmd_stop()
        if hasattr(d, "queued_cmd_clear"):
            d.queued_cmd_clear()
        if hasattr(d, "queued_cmd_start"):
            d.queued_cmd_start()
    except Exception as e:
        print("[DOBOT] queued cmd init failed:", e)

def dobot_connect():
    global dobot
    if not DOBOT_ENABLED:
        print("[DOBOT] disabled")
        return
    if Dobot is None:
        print("[DOBOT] pydobot not installed")
        return
    port = DOBOT_PORT or find_first_port()
    if not port:
        print("[DOBOT] no serial port found")
        return
    try:
        print(f"[DOBOT] connecting {port} ...")
        dobot = Dobot(port=port, verbose=False)
        _dobot_configure_motion(dobot)

        try:
            pose = dobot.pose()
            print("[DOBOT] connected, pose:", pose)
        except Exception as e:
            print("[DOBOT] connected but pose() failed:", e)

        if "home" in positions_4d:
            x, y, z, r = positions_4d["home"]
            move_xyzr(x, y, z + SAFE_Z, r, wait=True)
            move_xyzr(x, y, z, r, wait=True)

    except Exception as e:
        print("[DOBOT] connect failed:", e)
        dobot = None

def _ptp_move(x, y, z, r, wait=True):
    if hasattr(dobot, "move_to"):
        dobot.move_to(x, y, z, r, wait=wait)
        return
    mode = 2
    if hasattr(dobot, "set_ptp_cmd"):
        dobot.set_ptp_cmd(x, y, z, r, mode, wait)
    else:
        if hasattr(dobot, "send_ptp_cmd"):
            dobot.send_ptp_cmd(x, y, z, r, mode, wait)
        else:
            raise AttributeError("No valid PTP move method found in pydobot.")

def _within_workspace(x, y, z):
    return (-300 <= x <= 300) and (-300 <= y <= 300) and (-60 <= z <= 200)

def move_xyzr(x, y, z, r, wait=True):
    if not DOBOT_ENABLED or dobot is None:
        print(f"[DOBOT] move skipped: {(x,y,z,r)}")
        return
    if not _within_workspace(x, y, z):
        print(f"[DOBOT] target out of workspace: {(x,y,z,r)}  (SKIP)")
        return
    try:
        _ptp_move(x, y, z, r, wait=wait)
    except Exception as e:
        print("[DOBOT] move error:", e)

def suction(on=True):
    if not DOBOT_ENABLED or dobot is None:
        print(f"[DOBOT] suction {'ON' if on else 'OFF'} skipped")
        return
    try:
        if hasattr(dobot, "set_end_effector_suction_cup"):
            dobot.set_end_effector_suction_cup(enable=True, on=bool(on))
        elif hasattr(dobot, "suck"):
            dobot.suck(bool(on))
        else:
            print("[DOBOT] no suction API found in this pydobot")
    except Exception as e:
        print("[DOBOT] suction error:", e)

def safe_approach(x, y, z, r):
    move_xyzr(x, y, z + SAFE_Z, r, wait=True)
    move_xyzr(x, y, z, r, wait=True)

def safe_depart(x, y, z, r):
    move_xyzr(x, y, z + SAFE_Z, r, wait=True)

def pick_and_place(pick_name, place_name):
    global dobot_busy, last_job_time
    if dobot_busy:
        print("[DOBOT] busy, skip")
        return
    now = time.time()
    if now - last_job_time < TRIGGER_COOLDOWN:
        print("[DOBOT] cooldown, skip")
        return
    dobot_busy = True
    try:
        if pick_name not in positions_4d or place_name not in positions_4d:
            print(f"[DOBOT] invalid pos: {pick_name} -> {place_name}")
            return
        px, py, pz, pr = positions_4d[pick_name]
        qx, qy, qz, qr = positions_4d[place_name]

        print(f"[DOBOT] PICK {pick_name} -> PLACE {place_name}")
        safe_approach(px, py, pz, pr)
        suction(True)
        time.sleep(0.25)
        safe_depart(px, py, pz, pr)

        safe_approach(qx, qy, qz, qr)
        suction(False)
        time.sleep(0.25)
        safe_depart(qx, qy, qz, qr)

        if RETURN_HOME and "home" in positions_4d:
            hx, hy, hz, hr = positions_4d["home"]
            move_xyzr(hx, hy, hz + SAFE_Z, hr, wait=True)
            move_xyzr(hx, hy, hz, hr, wait=True)

        print("[DOBOT] job done.")
    except Exception as e:
        print("[DOBOT] error:", e)
    finally:
        last_job_time = time.time()
        dobot_busy = False

def get_position_name_from_char(char):
    if 'A' <= char <= 'P':
        return ("block_1", "drop_off_1")
    elif 'Q' <= char <= 'Y':
        return ("block_2", "drop_off_2")
    else:
        return (None, None)

# =================== ML-BASED ASL DETECTOR ===================
class ASLMLPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.labels = []
        self.load_model()
        
        # สำหรับ smoothing
        self.prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
        
    def load_model(self):
        """โหลดโมเดลที่ฝึกแล้ว"""
        if not os.path.exists(MODEL_FILE):
            print(f"[ML] ไม่พบโมเดล: {MODEL_FILE}")
            print("[ML] จะใช้ rule-based แทน")
            return
        
        try:
            self.model = joblib.load(MODEL_FILE)
            print("[ML] โหลดโมเดลสำเร็จ")
            
            # โหลดข้อมูลเสริม
            if os.path.exists(MODEL_INFO_FILE):
                with open(MODEL_INFO_FILE, 'r') as f:
                    model_info = json.load(f)
                    self.feature_columns = model_info.get('feature_columns', [])
                    self.labels = model_info.get('labels', [])
            
        except Exception as e:
            print(f"[ML] เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
            self.model = None
    
    def extract_hand_features(self, landmarks):
        """สกัดฟีเจอร์จาก hand landmarks (เหมือนกับตอนเทรน)"""
        features = []
        
        # 1. Raw coordinates (normalized)
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])
        
        # 2. Distance features
        distances = []
        thumb_tip = landmarks[4]
        for i in [8, 12, 16, 20]:  # index, middle, ring, pinky tips
            fingertip = landmarks[i]
            dist = np.sqrt((thumb_tip.x - fingertip.x)**2 + 
                          (thumb_tip.y - fingertip.y)**2 + 
                          (thumb_tip.z - fingertip.z)**2)
            distances.append(dist)
        
        # Distance between adjacent fingertips
        fingertips = [8, 12, 16, 20]
        for i in range(len(fingertips) - 1):
            p1 = landmarks[fingertips[i]]
            p2 = landmarks[fingertips[i + 1]]
            dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
            distances.append(dist)
        
        features.extend(distances)
        
        # 3. Angle features
        angles = []
        finger_joints = [
            [1, 2, 3, 4],    # thumb
            [5, 6, 7, 8],    # index
            [9, 10, 11, 12], # middle
            [13, 14, 15, 16],# ring
            [17, 18, 19, 20] # pinky
        ]
        
        for joints in finger_joints:
            for i in range(len(joints) - 2):
                p1 = landmarks[joints[i]]
                p2 = landmarks[joints[i + 1]]
                p3 = landmarks[joints[i + 2]]
                
                v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
                v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)
        
        features.extend(angles)
        
        # 4. Relative position features
        wrist = landmarks[0]
        relative_positions = []
        for i in [4, 8, 12, 16, 20]:  # fingertips
            tip = landmarks[i]
            rel_x = tip.x - wrist.x
            rel_y = tip.y - wrist.y
            rel_z = tip.z - wrist.z
            relative_positions.extend([rel_x, rel_y, rel_z])
        
        features.extend(relative_positions)
        
        return np.array(features)
    
    def predict(self, landmarks):
        """ทำนายตัวอักษรจาก landmarks"""
        if self.model is None:
            return "NO_MODEL", 0.0
        
        try:
            features = self.extract_hand_features(landmarks)
            features = features.reshape(1, -1)
            
            # ทำนาย
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities.max()
            
            # เพิ่มใน buffer สำหรับ smoothing
            self.prediction_buffer.append((prediction, confidence))
            
            return prediction, confidence
            
        except Exception as e:
            print(f"[ML] Prediction error: {e}")
            return "ERROR", 0.0
    
    def get_smoothed_prediction(self):
        """ได้ prediction ที่ smooth แล้วจาก buffer"""
        if not self.prediction_buffer:
            return "NONE", 0.0
        
        # หาตัวอักษรที่มี confidence สูงสุดจาก buffer ล่าสุด
        recent_predictions = list(self.prediction_buffer)[-5:]  # ดู 5 เฟรมล่าสุด
        
        # กรอง predictions ที่มี confidence สูงพอ
        valid_preds = [(pred, conf) for pred, conf in recent_predictions if conf >= MIN_CONFIDENCE]
        
        if not valid_preds:
            return "LOW_CONF", 0.0
        
        # หาตัวที่ปรากฏบ่อยที่สุด
        pred_counts = {}
        for pred, conf in valid_preds:
            if pred not in pred_counts:
                pred_counts[pred] = {'count': 0, 'total_conf': 0.0}
            pred_counts[pred]['count'] += 1
            pred_counts[pred]['total_conf'] += conf
        
        # เลือกตัวที่มี count มากที่สุด และ average confidence สูง
        best_pred = None
        best_score = 0
        
        for pred, data in pred_counts.items():
            avg_conf = data['total_conf'] / data['count']
            score = data['count'] * avg_conf  # รวมความบ่อยกับ confidence
            
            if score > best_score:
                best_score = score
                best_pred = pred
        
        if best_pred:
            avg_conf = pred_counts[best_pred]['total_conf'] / pred_counts[best_pred]['count']
            return best_pred, avg_conf
        
        return "UNCERTAIN", 0.0

# =================== FALLBACK RULE-BASED (simplified) ===================
def _v(a, b):
    return (b.x - a.x, b.y - a.y, b.z - a.z)

def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _norm(a):
    return math.sqrt(_dot(a, a)) + 1e-9

def _cos(a, b):
    return _dot(a, b) / (_norm(a) * _norm(b))

def _dist(p, q):
    return math.sqrt((p.x-q.x)**2 + (p.y-q.y)**2 + (p.z-q.z)**2)

def _is_extended(lm, tip, pip, mcp, cos_th=0.85):
    a = _v(lm[mcp], lm[pip])
    b = _v(lm[pip], lm[tip])
    return _cos(a, b) > cos_th

def classify_letter_simple(lm, handedness_label):
    """Rule-based classifier (simplified version เป็น fallback)"""
    I_TIP, I_PIP, I_MCP = lm[HL.INDEX_FINGER_TIP], lm[HL.INDEX_FINGER_PIP], lm[HL.INDEX_FINGER_MCP]
    M_TIP, M_PIP, M_MCP = lm[HL.MIDDLE_FINGER_TIP], lm[HL.MIDDLE_FINGER_PIP], lm[HL.MIDDLE_FINGER_MCP]
    R_TIP, R_PIP, R_MCP = lm[HL.RING_FINGER_TIP], lm[HL.RING_FINGER_PIP], lm[HL.RING_FINGER_MCP]
    P_TIP, P_PIP, P_MCP = lm[HL.PINKY_TIP], lm[HL.PINKY_PIP], lm[HL.PINKY_MCP]
    T_TIP, T_IP,  T_MCP = lm[HL.THUMB_TIP], lm[HL.THUMB_IP], lm[HL.THUMB_MCP]

    I_ext = _is_extended(lm, HL.INDEX_FINGER_TIP, HL.INDEX_FINGER_PIP, HL.INDEX_FINGER_MCP)
    M_ext = _is_extended(lm, HL.MIDDLE_FINGER_TIP, HL.MIDDLE_FINGER_PIP, HL.MIDDLE_FINGER_MCP)
    R_ext = _is_extended(lm, HL.RING_FINGER_TIP, HL.RING_FINGER_PIP, HL.RING_FINGER_MCP)
    P_ext = _is_extended(lm, HL.PINKY_TIP, HL.PINKY_PIP, HL.PINKY_MCP)

    # ตัวอย่างกฎเบื้องต้น
    if not I_ext and not M_ext and not R_ext and not P_ext:
        return "A"
    if I_ext and M_ext and R_ext and P_ext:
        return "B"
    if I_ext and not M_ext and not R_ext and not P_ext:
        return "D"
    if P_ext and not I_ext and not M_ext and not R_ext:
        return "I"
    if I_ext and M_ext and not R_ext and not P_ext:
        return "V"
    
    return "UNK"

# =================== MAIN ===================
def main():
    # เริ่มต้น
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("ไม่พบกล้อง")

    # เชื่อมต่อ Dobot
    dobot_connect()
    
    # สร้าง ML predictor
    ml_predictor = ASLMLPredictor()
    use_ml = ml_predictor.model is not None
    
    print(f"[SYSTEM] ใช้ {'ML Model' if use_ml else 'Rule-based'} สำหรับการจำแนก")

    # ตัวแปรสำหรับการยืนยันตัวอักษร
    last_letter = "-"
    hold = 0
    confirmed = ""
    current_pos_name = None

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        print("เริ่มจับภาพ — ยกมือทำท่าตัวอักษร")
        
        while True:
            ret, frame = cap.read()
            if not ret: 
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            letter = "-"
            confidence = 0.0
            
            if results.multi_hand_landmarks and results.multi_handedness:
                lms = results.multi_hand_landmarks[0]
                handed = results.multi_handedness[0].classification[0].label
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

                if use_ml:
                    # ใช้ ML prediction
                    pred, conf = ml_predictor.predict(lms.landmark)
                    letter, confidence = ml_predictor.get_smoothed_prediction()
                else:
                    # ใช้ rule-based fallback
                    letter = classify_letter_simple(lms.landmark, handed)
                    confidence = 0.8 if letter != "UNK" else 0.3

            # กลไก hold เพื่อยืนยัน (ต้องมี confidence สูงพอด้วย)
            if letter == last_letter and letter not in ("UNK", "-", "NO_MODEL", "ERROR", "LOW_CONF", "UNCERTAIN") and confidence >= MIN_CONFIDENCE:
                hold += 1
            else:
                hold = 0
                last_letter = letter

            # เมื่อยืนยันได้
            if hold >= HOLD_FRAMES and letter not in ("UNK", "-", "NO_MODEL", "ERROR", "LOW_CONF", "UNCERTAIN"):
                confirmed = letter
                hold = 0
                print(f"\n[CONFIRMED] ตัวอักษร: {confirmed} (confidence: {confidence:.2f})")

                # เล่นเสียง
                snd = ensure_sound_for_char(confirmed)
                if snd:
                    try:
                        snd.play()
                    except Exception as e:
                        print("Error playing sound:", e)

                # สั่ง Dobot
                start_name, end_name = get_position_name_from_char(confirmed)
                if start_name and end_name:
                    current_pos_name = start_name
                    print(f"[DOBOT] Moving to pick at {start_name}")
                    
                    pick_and_place(start_name, end_name)
                    
                    current_pos_name = end_name
                    print(f"[DOBOT] Placed at {end_name}")
                else:
                    print("[DOBOT] ไม่มีตำแหน่งแมปสำหรับตัวอักษรนี้")

            # UI Display
            h, w, _ = frame.shape
            
            # สถานะการจำแนก
            model_type = "ML" if use_ml else "Rule"
            cv2.putText(frame, f"Model: {model_type}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # ผลการทำนาย
            color = (0, 255, 0) if confidence >= MIN_CONFIDENCE else (0, 165, 255)
            cv2.putText(frame, f"Current: {letter} ({confidence:.2f})", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # สถานะการยืนยัน
            cv2.putText(frame, f"Hold: {hold}/{HOLD_FRAMES}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # ตัวอักษรที่ยืนยันแล้ว
            cv2.putText(frame, f"Confirmed: {confirmed}", (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # ตำแหน่งปัจจุบัน
            if current_pos_name:
                cv2.putText(frame, f"Position: {current_pos_name}", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # คำแนะนำ
            hint1 = "A–P: block_1->drop_off_1,  Q–Y: block_2->drop_off_2"
            hint2 = "Press: h=home, 1=test_A-P, 2=test_Q-Y, q=quit"
            cv2.putText(frame, hint1, (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, hint2, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("ASL ML Recognition + Dobot Control", frame)
            
            # คีย์บอร์ดควบคุม
            key = cv2.waitKey(1) & 0xFF
            if key == ord('h'):
                # ทดสอบกลับ home
                if "home" in positions_4d:
                    hx, hy, hz, hr = positions_4d["home"]
                    move_xyzr(hx, hy, hz + SAFE_Z, hr, wait=True)
                    move_xyzr(hx, hy, hz, hr, wait=True)
            elif key == ord('1'):
                # ทดสอบ A-P group
                pick_and_place("block_1", "drop_off_1")
            elif key == ord('2'):
                # ทดสอบ Q-Y group
                pick_and_place("block_2", "drop_off_2")
            elif key == ord('r'):
                # รีเซ็ต prediction buffer
                if use_ml:
                    ml_predictor.prediction_buffer.clear()
                print("[SYSTEM] Reset prediction buffer")
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # ปิด Dobot
    if dobot is not None:
        try: 
            dobot.suck(False)
        except Exception: 
            pass
        try: 
            dobot.disconnect()
        except Exception: 
            pass

if __name__ == "__main__":
    main()