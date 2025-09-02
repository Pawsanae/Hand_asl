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
import sys
from gtts import gTTS
import serial.tools.list_ports as list_ports
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

# Dobot settings - Simplified
DOBOT_ENABLED = True
SAFE_HEIGHT = 50.0
HOME_COORDS = (155.15, 58.87, 37.87, 20.78)

# Block and drop-off positions (x, y, z, r)
BLOCK_1_COORDS = (140.97, 126.50, -37.63, 41.90)
DROP_OFF_1_COORDS = (192.14, 147.91, -37.70, 7.59)
BLOCK_2_COORDS = (188.56, 14.84, -36.73, 4.50)
DROP_OFF_2_COORDS = (237.50, 38.63, -38.24, 9.24)

# ML prediction settings
PREDICTION_BUFFER_SIZE = 25
MIN_CONFIDENCE = 0.5
HOLD_FRAMES = 8
TRIGGER_COOLDOWN = 2.0

# Global variables
dobot = None
enable = True
disable = False
command_in_progress = False

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
    """สร้างและโหลดเสียงสำหรับตัวอักษร"""
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

# Pre-load all sounds
for ch in supported_chars:
    ensure_sound_for_char(ch)

# ------- Simplified Dobot Functions -------
def connect_dobot():
    """เชื่อมต่อกับ Dobot แบบง่าย"""
    global dobot, enable, disable
    
    if not DOBOT_ENABLED:
        print("[DOBOT] disabled")
        return
    
    print('Connecting to Dobot...')
    ports = list(list_ports.comports())
    print(f"Found {len(ports)} serial ports:")
    for port in ports:
        print(f"  - {port.device}: {port.description}")
    
    # ลองหาพอร์ตอัตโนมัติ
    serial_port = '/dev/ttyUSB0'
    for port in ports:
        if 'USB' in port.device or 'ACM' in port.device:
            serial_port = port.device
            print(f"Auto-selected port: {serial_port}")
            break
    
    try:
        print(f"Trying to connect to {serial_port}...")
        dobot = Dobot(port=serial_port, verbose=True)
        
        # ตรวจสอบสถานะเริ่มต้น
        try:
            current_pose = dobot.pose()
            print(f"Current Dobot position: {current_pose}")
        except Exception as e:
            print(f"Warning: Could not get current pose: {e}")
        
        # ตั้งค่าเริ่มต้น
        print("Turning off suction...")
        dobot.suck(disable)
        dobot.wait(500)
        
        print(f"Moving to HOME position: {HOME_COORDS}...")
        dobot.move_to(*HOME_COORDS)
        dobot.wait(2000)  # รอให้เคลื่อนที่เสร็จ
        
        # ตรวจสอบว่าถึง HOME แล้ว
        try:
            final_pose = dobot.pose()
            print(f"Final position after HOME: {final_pose}")
        except Exception as e:
            print(f"Warning: Could not verify final pose: {e}")
        
        print("Dobot connected and at HOME position.")
        return True
        
    except Exception as e:
        print(f"Error connecting to Dobot: {e}")
        print("Troubleshooting tips:")
        print("1. Check if Dobot is powered on")
        print("2. Check USB connection")
        print("3. Try different USB port")
        print("4. Check if another program is using the port")
        dobot = None
        return False

def pick_and_place(pickup_coords, dropoff_coords):
    """ฟังก์ชัน pick and place แบบง่าย"""
    global command_in_progress
    
    if not DOBOT_ENABLED or dobot is None:
        print("[DOBOT] Not connected, skipping pick and place")
        return
    
    if command_in_progress:
        print("[DOBOT] Command in progress, skipping...")
        return
    
    command_in_progress = True
    
    try:
        print(f"Moving to pick up block at {pickup_coords} and place at {dropoff_coords}")
        
        # 1. Move to a safe height above the block
        dobot.move_to(pickup_coords[0], pickup_coords[1], SAFE_HEIGHT, pickup_coords[3])
        dobot.wait(500)
        
        # 2. Lower the arm to pick up the block
        dobot.move_to(pickup_coords[0], pickup_coords[1], pickup_coords[2], pickup_coords[3])
        dobot.suck(enable)  # Turn on the suction cup
        dobot.wait(1000)
        
        # 3. Lift the arm back to the safe height
        dobot.move_to(pickup_coords[0], pickup_coords[1], SAFE_HEIGHT, pickup_coords[3])
        dobot.wait(500)
        
        print(f"Moving to drop off block at {dropoff_coords}")
        
        # 4. Move to a safe height above the drop-off location
        dobot.move_to(dropoff_coords[0], dropoff_coords[1], SAFE_HEIGHT, dropoff_coords[3])
        dobot.wait(500)
        
        # 5. Lower the arm to drop off the block
        dobot.move_to(dropoff_coords[0], dropoff_coords[1], dropoff_coords[2], dropoff_coords[3])
        dobot.suck(disable)  # Turn off the suction cup
        dobot.wait(1000)
        
        # 6. Lift the arm back to the safe height
        dobot.move_to(dropoff_coords[0], dropoff_coords[1], SAFE_HEIGHT, dropoff_coords[3])
        dobot.wait(500)
        
        # 7. Return to HOME position
        dobot.move_to(*HOME_COORDS)
        dobot.wait(1000)
        
        print("Pick and place complete. Returned to Home position.")
        
    except Exception as e:
        print(f"[DOBOT] Pick and place error: {e}")
    
    finally:
        # Reset the flag to allow new commands
        command_in_progress = False

def test_dobot_movement():
    """ทดสอบการเคลื่อนที่ของ Dobot"""
    if dobot is None:
        print("[TEST] Dobot not connected")
        return
    
    try:
        print("[TEST] Testing Dobot movement...")
        
        # ตรวจสอบตำแหน่งปัจจุบัน
        try:
            current = dobot.pose()
            print(f"[TEST] Current position: {current}")
        except Exception as e:
            print(f"[TEST] Could not get current position: {e}")
        
        # ทดสอบการเคลื่อนที่เล็กน้อย
        test_coords = [
            (180, 0, 50, 0),    # ขวา
            (120, 0, 50, 0),    # ซ้าย
            HOME_COORDS         # กลับ HOME
        ]
        
        for i, coords in enumerate(test_coords):
            print(f"[TEST] Step {i+1}: Moving to {coords}")
            dobot.move_to(*coords)
            dobot.wait(2000)
            
            # ตรวจสอบตำแหน่งหลังเคลื่อนที่
            try:
                actual = dobot.pose()
                print(f"[TEST] Reached: {actual}")
            except Exception:
                pass
        
        print("[TEST] Movement test completed successfully")
        
    except Exception as e:
        print(f"[TEST] Error during movement test: {e}")

def test_suction():
    """ทดสอบระบบดูด"""
    if dobot is None:
        print("[TEST] Dobot not connected")
        return
    
    try:
        print("[TEST] Testing suction system...")
        
        print("[TEST] Turning suction ON...")
        dobot.suck(enable)
        dobot.wait(2000)
        
        print("[TEST] Turning suction OFF...")
        dobot.suck(disable)
        dobot.wait(2000)
        
        print("[TEST] Suction test completed")
        
    except Exception as e:
        print(f"[TEST] Error during suction test: {e}")

def get_coords_from_char(char):
    """ได้พิกัดสำหรับตัวอักษร"""
    if 'A' <= char <= 'P':
        return (BLOCK_1_COORDS, DROP_OFF_1_COORDS)
    elif 'Q' <= char <= 'Y':
        return (BLOCK_2_COORDS, DROP_OFF_2_COORDS)
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
            print(f"[ML] Model not found: {MODEL_FILE}")
            print("[ML] Will use rule-based classifier instead")
            return
        
        try:
            self.model = joblib.load(MODEL_FILE)
            print("[ML] Model loaded successfully")
            
            # โหลดข้อมูลเสริม
            if os.path.exists(MODEL_INFO_FILE):
                with open(MODEL_INFO_FILE, 'r') as f:
                    model_info = json.load(f)
                    self.feature_columns = model_info.get('feature_columns', [])
                    self.labels = model_info.get('labels', [])
            
        except Exception as e:
            print(f"[ML] Error loading model: {e}")
            self.model = None
    
    def extract_hand_features(self, landmarks):
        """สกัดฟีเจอร์จาก hand landmarks"""
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
        recent_predictions = list(self.prediction_buffer)[-5:]
        
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
            score = data['count'] * avg_conf
            
            if score > best_score:
                best_score = score
                best_pred = pred
        
        if best_pred:
            avg_conf = pred_counts[best_pred]['total_conf'] / pred_counts[best_pred]['count']
            return best_pred, avg_conf
        
        return "UNCERTAIN", 0.0

# =================== FALLBACK RULE-BASED ===================
def _v(a, b):
    return (b.x - a.x, b.y - a.y, b.z - a.z)

def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _norm(a):
    return math.sqrt(_dot(a, a)) + 1e-9

def _cos(a, b):
    return _dot(a, b) / (_norm(a) * _norm(b))

def _is_extended(lm, tip, pip, mcp, cos_th=0.85):
    a = _v(lm[mcp], lm[pip])
    b = _v(lm[pip], lm[tip])
    return _cos(a, b) > cos_th

def classify_letter_simple(lm, handedness_label):
    """Rule-based classifier (simplified fallback)"""
    I_ext = _is_extended(lm, HL.INDEX_FINGER_TIP, HL.INDEX_FINGER_PIP, HL.INDEX_FINGER_MCP)
    M_ext = _is_extended(lm, HL.MIDDLE_FINGER_TIP, HL.MIDDLE_FINGER_PIP, HL.MIDDLE_FINGER_MCP)
    R_ext = _is_extended(lm, HL.RING_FINGER_TIP, HL.RING_FINGER_PIP, HL.RING_FINGER_MCP)
    P_ext = _is_extended(lm, HL.PINKY_TIP, HL.PINKY_PIP, HL.PINKY_MCP)

    # กฎเบื้องต้น
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
        raise RuntimeError("Camera not found")

    # เชื่อมต่อ Dobot
    connect_dobot()
    
    # สร้าง ML predictor
    ml_predictor = ASLMLPredictor()
    use_ml = ml_predictor.model is not None
    
    print(f"[SYSTEM] Using {'ML Model' if use_ml else 'Rule-based'} for classification")

    # ตัวแปรสำหรับการยืนยันตัวอักษร
    last_letter = "-"
    hold = 0
    confirmed = ""
    last_action_time = 0

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        print("Starting capture — Raise hand to make letter gestures")
        
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

            # กลไก hold เพื่อยืนยัน
            if (letter == last_letter and 
                letter not in ("UNK", "-", "NO_MODEL", "ERROR", "LOW_CONF", "UNCERTAIN") and 
                confidence >= MIN_CONFIDENCE):
                hold += 1
            else:
                hold = 0
                last_letter = letter

            # เมื่อยืนยันได้และไม่อยู่ในช่วง cooldown
            current_time = time.time()
            if (hold >= HOLD_FRAMES and 
                letter not in ("UNK", "-", "NO_MODEL", "ERROR", "LOW_CONF", "UNCERTAIN") and
                current_time - last_action_time > TRIGGER_COOLDOWN):
                
                confirmed = letter
                hold = 0
                last_action_time = current_time
                
                print(f"\n[CONFIRMED] Letter: {confirmed} (confidence: {confidence:.2f})")

                # เล่นเสียง
                snd = ensure_sound_for_char(confirmed)
                if snd:
                    try:
                        snd.play()
                    except Exception as e:
                        print("Error playing sound:", e)

                # สั่ง Dobot
                pickup_coords, dropoff_coords = get_coords_from_char(confirmed)
                if pickup_coords and dropoff_coords:
                    print(f"[DOBOT] Executing pick and place for letter {confirmed}")
                    pick_and_place(pickup_coords, dropoff_coords)
                else:
                    print(f"[DOBOT] No position mapping for letter {confirmed}")

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
            
            # สถานะ Dobot
            dobot_status = "Connected" if dobot else "Disconnected"
            dobot_color = (0, 255, 0) if dobot else (0, 0, 255)
            cv2.putText(frame, f"Dobot: {dobot_status}", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, dobot_color, 2)
            
            # คำแนะนำ
            hints = [
                "A-P: Block1->Drop1,  Q-Y: Block2->Drop2",
                "Keys: h=home, t=test_move, s=test_suction, p=show_pos",
                "      1=test A-P, 2=test Q-Y, r=reset, q=quit"
            ]
            for i, hint in enumerate(hints):
                cv2.putText(frame, hint, (10, h-65+i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            cv2.imshow("ASL ML Recognition + Dobot Control", frame)
            
            # คีย์บอร์ดควบคุม
            key = cv2.waitKey(1) & 0xFF
            if key == ord('h'):
                # กลับ home
                if dobot:
                    print("[MANUAL] Moving to HOME")
                    dobot.move_to(*HOME_COORDS)
                    dobot.wait(1000)
            elif key == ord('t'):
                # ทดสอบการเคลื่อนที่
                test_dobot_movement()
            elif key == ord('s'):
                # ทดสอบระบบดูด
                test_suction()
            elif key == ord('p'):
                # แสดงตำแหน่งปัจจุบัน
                if dobot:
                    try:
                        pos = dobot.pose()
                        print(f"[INFO] Current Dobot position: {pos}")
                    except Exception as e:
                        print(f"[INFO] Could not get position: {e}")
            elif key == ord('1'):
                # ทดสอบ A-P group
                if not command_in_progress:
                    pick_and_place(BLOCK_1_COORDS, DROP_OFF_1_COORDS)
            elif key == ord('2'):
                # ทดสอบ Q-Y group
                if not command_in_progress:
                    pick_and_place(BLOCK_2_COORDS, DROP_OFF_2_COORDS)
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
            dobot.suck(disable)
        except Exception: 
            pass
        try: 
            dobot.close()
        except Exception: 
            pass
        print("Dobot disconnected")

if __name__ == "__main__":
    main()
    