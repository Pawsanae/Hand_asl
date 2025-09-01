#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 1. 📊 เก็บข้อมูลฝึกแบบต่อเนื่อง
# 2. 🧠 ฝึกโมเดล
# 3. 🎯 ทดสอบ real-time
# 4. 📈 แสดงสถิติข้อมูล
# 5. 💾 จัดการไฟล์สำรอง
# 6. 🚪 ออกจากโปรแกรม

import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
from collections import deque
import time
from datetime import datetime

# ====== CONFIG ======
DATA_DIR = "asl_training_data"
MODEL_DIR = "trained_models"
FEATURES_FILE = os.path.join(DATA_DIR, "hand_features.csv")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")
MODEL_FILE = os.path.join(MODEL_DIR, "asl_classifier.pkl")
MODEL_INFO_FILE = os.path.join(MODEL_DIR, "model_info.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ชุดตัวอักษร
LETTERS = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

# ====== ฟังก์ชันสกัดฟีเจอร์ (ใช้ร่วมกันระหว่างเก็บข้อมูล/เทรน/พรีดิกต์) ======
def extract_hand_features(landmarks):
    """รับ list[21] ของ mediapipe NormalizedLandmark -> np.array shape (n_features,)"""
    feats = []

    # 1) raw normalized xyz (21 * 3)
    for lm in landmarks:
        feats.extend([lm.x, lm.y, lm.z])

    # 2) distances: thumb tip (4) กับนิ้วอื่น (8,12,16,20) + ระยะระหว่างปลายนิ้วติดกัน (8-12, 12-16, 16-20)
    thumb_tip = landmarks[4]
    for i in [8, 12, 16, 20]:
        p = landmarks[i]
        d = np.sqrt((thumb_tip.x - p.x)**2 + (thumb_tip.y - p.y)**2 + (thumb_tip.z - p.z)**2)
        feats.append(d)
    for i1, i2 in [(8,12),(12,16),(16,20)]:
        p1, p2 = landmarks[i1], landmarks[i2]
        d = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
        feats.append(d)

    # 3) angles ตามข้อนิ้ว (นิ้วละ 3 มุม ยกเว้นหัวแม่มือก็ 3)
    angles = []
    finger_joints = [
        [1,2,3,4],      # thumb
        [5,6,7,8],      # index
        [9,10,11,12],   # middle
        [13,14,15,16],  # ring
        [17,18,19,20],  # pinky
    ]
    for js in finger_joints:
        for i in range(len(js)-2):
            p1, p2, p3 = landmarks[js[i]], landmarks[js[i+1]], landmarks[js[i+2]]
            v1 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
            v2 = np.array([p3.x-p2.x, p3.y-p2.y, p3.z-p2.z])
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cosang = float(np.dot(v1, v2) / denom)
            angles.append(np.arccos(np.clip(cosang, -1.0, 1.0)))
    feats.extend(angles)

    # 4) relative to wrist (0) เฉพาะปลายนิ้ว [4,8,12,16,20]
    wrist = landmarks[0]
    for i in [4,8,12,16,20]:
        tip = landmarks[i]
        feats.extend([tip.x - wrist.x, tip.y - wrist.y, tip.z - wrist.z])

    return np.array(feats, dtype=np.float32)

# ====== เก็บข้อมูล ======
class ASLDataCollector:
    def __init__(self):
        self.features = []
        self.labels = []
        self.load_existing_data()

    def load_existing_data(self):
        if os.path.exists(FEATURES_FILE):
            try:
                df = pd.read_csv(FEATURES_FILE)
                print(f"พบข้อมูลเดิม: {len(df)} ตัวอย่าง")
                counts = df['label'].value_counts().sort_index()
                for k, v in counts.items():
                    print(f"  {k}: {v} ตัวอย่าง")
                feat_cols = [c for c in df.columns if c != 'label']
                self.features = df[feat_cols].values.tolist()
                self.labels = df['label'].tolist()
            except Exception as e:
                print(f"โหลดข้อมูลเดิมผิดพลาด: {e}")
                self.features, self.labels = [], []
        else:
            print("ไม่พบข้อมูลเดิม - เริ่มเก็บใหม่")

    def backup_data(self):
        if os.path.exists(FEATURES_FILE):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = os.path.join(BACKUP_DIR, f"hand_features_backup_{ts}.csv")
            try:
                import shutil
                shutil.copy2(FEATURES_FILE, dst)
                print(f"สำรองข้อมูลแล้ว: {dst}")
            except Exception as e:
                print(f"สำรองข้อมูลล้มเหลว: {e}")

    def get_current_count(self, letter): return self.labels.count(letter) if self.labels else 0

    def show_current_stats(self):
        if not self.labels:
            print("ยังไม่มีข้อมูลในระบบ"); return
        print(f"\n=== สถิติข้อมูลปัจจุบัน (รวม {len(self.labels)} ตัวอย่าง) ===")
        from collections import Counter
        cnt = Counter(self.labels)
        for letter in sorted(LETTERS):
            print(f"  {letter}: {cnt.get(letter,0):3d} ตัวอย่าง")

    def save_data(self, auto_save=False):
        if not self.features:
            if not auto_save: print("ไม่มีข้อมูลให้บันทึก")
            return
        try:
            X = np.asarray(self.features, dtype=np.float32)
            y = np.asarray(self.labels, dtype=object)
            cols = [f"feature_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols); df['label'] = y
            df.to_csv(FEATURES_FILE, index=False)
            if not auto_save:
                print(f"บันทึกข้อมูลแล้ว: {FEATURES_FILE} ({len(df)} ตัวอย่าง)")
        except Exception as e:
            print(f"บันทึกข้อมูลผิดพลาด: {e}")

    def collect_data_continuous(self, samples_per_session=50, cam_index=0):
        self.show_current_stats()
        print("\n=== โหมดเก็บข้อมูลต่อเนื่อง ===")
        print("0. เก็บทุกตัวอักษรตามลำดับ")
        for i, L in enumerate(LETTERS, 1):
            print(f"{i:2d}. {L} (ปัจจุบัน: {self.get_current_count(L)})")
        choice = input("\nเลือกตัวอักษร (0 สำหรับทั้งหมด, หรือหมายเลข): ").strip()

        letters_to_collect = LETTERS if choice == '0' else []
        if not letters_to_collect:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(LETTERS): letters_to_collect = [LETTERS[idx]]
                else: print("หมายเลขไม่ถูกต้อง"); return
            except: print("กรุณาป้อนหมายเลข"); return

        self.backup_data()

        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("❌ เปิดกล้องไม่ได้"); return

        with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,          # ใส่ 0 ได้ถ้าอยากให้เร็วขึ้น (ความละเอียดโมเดลต่ำลง)
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            ) as hands:
            for letter in letters_to_collect:
                print(f"\n=== {letter} === เตรียมท่าแล้วกด SPACE เพื่อเริ่ม/หยุด | S=ข้าม | N=ถัดไป | Q=ออก")
                collected_this = 0
                collecting = False
                stable = 0; required_stable = 5

                while collected_this < samples_per_session:
                    ok, frame = cap.read()
                    if not ok: continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands.process(rgb)

                    cv2.putText(frame, f"Letter: {letter}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                    cv2.putText(frame, f"Total:{self.get_current_count(letter)}  Session:{collected_this}/{samples_per_session}",
                                (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(frame, "SPACE=start/stop  S=skip  N=next  Q=quit",
                                (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                    if res.multi_hand_landmarks:
                        lms = res.multi_hand_landmarks[0]
                        mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
                        if collecting:
                            stable += 1
                            if stable >= required_stable:
                                f = extract_hand_features(lms.landmark)
                                self.features.append(f.tolist())
                                self.labels.append(letter)
                                collected_this += 1
                                stable = 0
                                if collected_this % 10 == 0: self.save_data(auto_save=True)
                    else:
                        if collecting:
                            cv2.putText(frame, "No hand detected!", (10,120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                    if collecting:
                        cv2.putText(frame, "COLLECTING...", (10,110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    else:
                        cv2.putText(frame, "Press SPACE to start", (10,110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

                    cv2.imshow("ASL Continuous Data Collection", frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord(' '):
                        collecting = not collecting; stable = 0
                        print(("เริ่ม" if collecting else "หยุด") + "เก็บข้อมูล")
                    elif k == ord('s'):
                        print(f"ข้าม {letter}"); break
                    elif k == ord('n'):
                        print(f"ถัดไป (ได้ {collected_this})"); break
                    elif k == ord('q'):
                        cap.release(); cv2.destroyAllWindows(); self.save_data(); return

                print(f"เสร็จ {letter}: เพิ่ม {collected_this}")
                self.save_data()

        cap.release()
        cv2.destroyAllWindows()
        self.save_data()
        print("\n=== เก็บข้อมูลเสร็จสิ้น ===")
        self.show_current_stats()

# ====== เทรนโมเดล ======
class ASLModelTrainer:
    def __init__(self):
        self.model = None
        self.feature_columns = None

    def load_data(self):
        if not os.path.exists(FEATURES_FILE):
            print(f"ไม่พบไฟล์ข้อมูล: {FEATURES_FILE}"); return None, None
        df = pd.read_csv(FEATURES_FILE)
        print(f"โหลดข้อมูล: {len(df)} ตัวอย่าง")
        counts = df['label'].value_counts().sort_index()
        for k,v in counts.items(): print(f"  {k}: {v}")
        min_samples = 20
        low = counts[counts < min_samples]
        if len(low) > 0:
            print(f"\n⚠️ ควรเก็บเพิ่ม (<{min_samples}): {', '.join(low.index.tolist())}")

        X = df.drop(columns=['label'])
        y = df['label']
        self.feature_columns = X.columns.tolist()
        return X.values.astype(np.float32), y.values

    def train_model(self):
        X, y = self.load_data()
        if X is None: return
        uniq = np.unique(y)
        if min((y==u).sum() for u in uniq) < 2:
            print("❌ ข้อมูลไม่พอ (อย่างน้อย 2 ตัวอย่างต่อคลาส)"); return

        test_size = min(0.2, 0.5 * min((y==u).sum() for u in uniq) / len(X))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        print(f"train: {len(Xtr)}  test: {len(Xte)}")

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
        self.model.fit(Xtr, ytr)
        yp = self.model.predict(Xte)
        acc = accuracy_score(yte, yp)
        print(f"\n✅ Accuracy: {acc:.3f}")
        print("\nClassification report:\n", classification_report(yte, yp))

        self.save_model()

        fi = pd.DataFrame({
            "feature":[f"feature_{i}" for i in range(len(self.model.feature_importances_))],
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        print("\nTop-10 features:\n", fi.head(10))

    def save_model(self):
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(self.model, MODEL_FILE)
            info = {
                "feature_columns": self.feature_columns,
                "labels": LETTERS,
                "trained_at": datetime.now().isoformat(),
                "n_features": len(self.feature_columns)
            }
            with open(MODEL_INFO_FILE, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            print(f"✅ บันทึกโมเดล: {MODEL_FILE}")
        except Exception as e:
            print(f"❌ บันทึกโมเดลล้มเหลว: {e}")

# ====== พรีดิกต์แบบเรียลไทม์ ======
class ASLRealTimePredictor:
    def __init__(self, cam_index=0):
        self.model = None
        self.n_features = None
        self.prediction_buffer = deque(maxlen=10)
        self.cam_index = cam_index
        self.load_model()

    def load_model(self):
        if not os.path.exists(MODEL_FILE):
            print(f"❌ ไม่พบโมเดล: {MODEL_FILE}"); return
        try:
            self.model = joblib.load(MODEL_FILE)
            if os.path.exists(MODEL_INFO_FILE):
                with open(MODEL_INFO_FILE, "r", encoding="utf-8") as f:
                    info = json.load(f)
                self.n_features = info.get("n_features", None)
                print(f"โมเดลฝึกเมื่อ: {info.get('trained_at','-')}, n_features={self.n_features}")
            print("✅ โหลดโมเดลเสร็จ")
        except Exception as e:
            print(f"❌ โหลดโมเดลผิดพลาด: {e}")

    def predict(self, landmarks):
        if self.model is None: return "NO_MODEL", 0.0
        f = extract_hand_features(landmarks).reshape(1, -1)
        if self.n_features is not None and f.shape[1] != self.n_features:
            # กัน shape mismatch
            print(f"⚠️ shape ฟีเจอร์ไม่ตรง: got {f.shape[1]} vs expected {self.n_features}")
            return "SHAPE_MISMATCH", 0.0
        try:
            probs = self.model.predict_proba(f)[0]
            pred_idx = int(np.argmax(probs))
            pred = self.model.classes_[pred_idx]
            conf = float(probs[pred_idx])
            return pred, conf
        except Exception as e:
            print(f"พรีดิกต์ผิดพลาด: {e}")
            return "ERROR", 0.0

    def run_realtime(self):
        if self.model is None:
            print("❌ ไม่มีโมเดลสำหรับทดสอบ"); return
        print("🚀 เริ่ม real-time | Q=ออก, R=รีเซ็ตบัฟเฟอร์")

        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            print("❌ เปิดกล้องไม่ได้"); return

        try:
            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,          # หรือ 0 ถ้าอยากให้เร็วขึ้น
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            ) as hands:
                while True:
                    ok, frame = cap.read()
                    if not ok: continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands.process(rgb)

                    prediction, confidence = "No Hand", 0.0
                    if res.multi_hand_landmarks:
                        lms = res.multi_hand_landmarks[0]
                        mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
                        pred, conf = self.predict(lms.landmark)
                        if conf > 0.3: self.prediction_buffer.append((pred, conf))
                        if self.prediction_buffer:
                            prediction, confidence = max(self.prediction_buffer, key=lambda x: x[1])

                    color = (0,255,0) if confidence>0.7 else (0,165,255) if confidence>0.4 else (0,0,255)
                    cv2.putText(frame, f"Prediction: {prediction}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f"Buffer: {len(self.prediction_buffer)}/10", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame, "Q=quit, R=reset buffer", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                    cv2.imshow("ASL Real-time Prediction", frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'): break
                    elif k == ord('r'):
                        self.prediction_buffer.clear(); print("รีเซ็ต buffer แล้ว")
        finally:
            cap.release()
            cv2.destroyAllWindows()

# ====== ยูทิลดูสถิติ/แบ็คอัพ ======
def show_detailed_stats():
    if not os.path.exists(FEATURES_FILE):
        print("❌ ไม่พบข้อมูล"); return
    df = pd.read_csv(FEATURES_FILE)
    print(f"\n📊 สถิติ\nไฟล์: {FEATURES_FILE}")
    print(f"ขนาด: {os.path.getsize(FEATURES_FILE)/(1024*1024):.2f} MB")
    print(f"ตัวอย่าง: {len(df)}  ฟีเจอร์/ตัวอย่าง: {len(df.columns)-1}")
    counts = df['label'].value_counts().sort_index()
    total = len(df)
    print("\nการกระจายต่อคลาส:")
    for L in sorted(LETTERS):
        c = int(counts.get(L,0)); pct = (c/total*100) if total else 0
        status = "✅" if c>=50 else "⚠️" if c>=20 else "❌"
        print(f"  {status} {L}: {c:3d} ({pct:5.1f}%)")
    low = counts[counts<50]
    if len(low): print("แนะนำเก็บเพิ่ม:", ", ".join(low.index))

def manage_backups():
    if not os.path.exists(BACKUP_DIR):
        print("❌ ไม่พบโฟลเดอร์สำรอง"); return
    files = [f for f in os.listdir(BACKUP_DIR) if f.endswith(".csv")]
    if not files:
        print("❌ ไม่พบไฟล์สำรอง"); return
    files.sort(reverse=True)
    print(f"\nไฟล์สำรอง ({len(files)}):")
    for i, fn in enumerate(files, 1):
        path = os.path.join(BACKUP_DIR, fn)
        size = os.path.getsize(path)/(1024*1024)
        print(f"{i:2d}. {fn}  [{size:.2f} MB]")

    print("\n1) คืนค่า  2) ลบไฟล์เก่า  3) กลับ")
    ch = input("เลือก (1-3): ").strip()
    if ch == '1':
        try:
            k = int(input("หมายเลขไฟล์: ")) - 1
            if 0 <= k < len(files):
                src = os.path.join(BACKUP_DIR, files[k])
                # สำรองปัจจุบัน
                if os.path.exists(FEATURES_FILE):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cur_bak = os.path.join(BACKUP_DIR, f"hand_features_before_restore_{ts}.csv")
                    import shutil; shutil.copy2(FEATURES_FILE, cur_bak)
                    print("สำรองไฟล์ปัจจุบันแล้ว:", cur_bak)
                import shutil; shutil.copy2(src, FEATURES_FILE)
                print("คืนค่าเรียบร้อย")
            else:
                print("หมายเลขไม่ถูกต้อง")
        except:
            print("กรุณากรอกเป็นตัวเลข")
    elif ch == '2':
        keep = input("เก็บล่าสุดกี่ไฟล์? (default 5): ").strip()
        try: keep = int(keep) if keep else 5
        except: keep = 5
        to_del = files[keep:] if len(files)>keep else []
        if not to_del: print("ไม่มีไฟล์เกินโควต้า"); return
        print("จะลบ:", ", ".join(to_del))
        if input("ยืนยัน (y/N): ").lower()=='y':
            n=0
            for f in to_del:
                try: os.remove(os.path.join(BACKUP_DIR,f)); n+=1
                except Exception as e: print("ลบไม่ได้:", f, e)
            print(f"ลบแล้ว {n} ไฟล์")
    else:
        return

# ====== เมนูหลัก ======
def main():
    print("="*50)
    print("🤖 ASL Machine Learning Training System (Continuous)")
    print("="*50)
    print("1. 📊 เก็บข้อมูลฝึกแบบต่อเนื่อง")
    print("2. 🧠 ฝึกโมเดล")
    print("3. 🎯 ทดสอบ real-time")
    print("4. 📈 แสดงสถิติข้อมูล")
    print("5. 💾 จัดการไฟล์สำรอง")
    print("6. 🚪 ออกจากโปรแกรม")

    collector = ASLDataCollector()

    while True:
        print("\n" + "="*30); collector.show_current_stats(); print("="*30)
        choice = input("\nเลือกตัวเลือก (1-6): ").strip()
        if choice == '1':
            samples = input("จำนวนตัวอย่างต่อ session (default: 50): ").strip()
            try:
                n = int(samples) if samples else 50
                if n <= 0: print("❌ ต้อง > 0"); continue
            except: n = 50
            collector = ASLDataCollector()
            collector.collect_data_continuous(n)
        elif choice == '2':
            ASLModelTrainer().train_model()
        elif choice == '3':
            ASLRealTimePredictor().run_realtime()
        elif choice == '4':
            show_detailed_stats()
        elif choice == '5':
            manage_backups()
        elif choice == '6':
            print("ขอบคุณที่ใช้ ASL Training System!"); break
        else:
            print("❌ กรุณาเลือก 1-6")

if __name__ == "__main__":
    main()
