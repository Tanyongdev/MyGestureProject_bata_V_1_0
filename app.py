import cv2
import numpy as np
import mediapipe as mp
import keras
from keras import layers
import joblib

SCALER_PATH  = "model/scaler.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
WEIGHTS_PATH = "model/MyGestureProject_bata_V1_0.h5"

CONF_THRESHOLD = 0.65

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def build_model():
    """สร้างโครงสร้างโมเดลตรงๆ ไม่ผ่าน config"""
    model = keras.Sequential([
        layers.Input(shape=(48,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64,  activation="relu"),
        layers.Dense(19,  activation="softmax"),
    ])
    return model


def extract_features(hand_landmarks, handedness):
    """
    สร้าง 48 Features
    bbox_x, bbox_y, bbox_w, bbox_h = 4
    lm_0 ... lm_41                 = 42
    leading_hand_left/right        = 2
    รวม                            = 48
    """
    features = []

    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    bbox_x = min(xs)
    bbox_y = min(ys)
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)

    features.extend([bbox_x, bbox_y, bbox_w, bbox_h])

    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y])

    if handedness == "Left":
        features.extend([0, 1])
    else:
        features.extend([1, 0])

    return np.array(features, dtype=np.float32)


def get_bounding_box(hand_landmarks, frame_w, frame_h, padding=20):
    xs = [lm.x * frame_w for lm in hand_landmarks.landmark]
    ys = [lm.y * frame_h for lm in hand_landmarks.landmark]

    x1 = max(0,        int(min(xs)) - padding)
    y1 = max(0,        int(min(ys)) - padding)
    x2 = min(frame_w,  int(max(xs)) + padding)
    y2 = min(frame_h,  int(max(ys)) + padding)

    return x1, y1, x2, y2


def run(camera_index=0):

    print("โหลดโมเดล...")
    model = build_model()
    model.load_weights(WEIGHTS_PATH)

    scaler = joblib.load(SCALER_PATH)
    le     = joblib.load(ENCODER_PATH)

    print("โหลดสำเร็จ!")
    
    cap = cv2.VideoCapture(camera_index)
    
    # 🛠️ ปรับสเปค 1: บีบความละเอียดกล้องลงมาเป็น 640x480 (ลื่นขึ้นเยอะ คอมไม่ต้องแบกภาพใหญ่)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ตัวแปรสำหรับทำ Frame Skipping
    frame_count = 0
    cached_results = [] # เก็บค่าทายรอบล่าสุดเอาไว้ใช้วาดในเฟรมที่ข้าม

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)
            
            # เพิ่มรอบนับเฟรม
            frame_count += 1

            if results.multi_hand_landmarks:
                
                # สร้างลิสต์ชั่วคราวเก็บค่าในเฟรมนี้
                current_frame_data = []

                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[i].classification[0].label
                    
                    # 🛠️ ปรับสเปค 2: สั่งให้โมเดลพยากรณ์ "ทุกๆ 3 เฟรม" (เฟรมอื่นใช้ค่าเดิม)
                    if frame_count % 5 == 0 or not cached_results or i >= len(cached_results):
                        # สั่งโมเดลทำงาน (กินสเปคช็อตนี้)
                        features    = extract_features(hand_landmarks, handedness)
                        inp         = scaler.transform(features.reshape(1, -1))
                        predictions = model.predict(inp, verbose=0)[0]

                        best_idx  = int(np.argmax(predictions))
                        best_conf = float(predictions[best_idx])

                        if best_conf >= CONF_THRESHOLD:
                            gesture = le.classes_[best_idx]
                        else:
                            gesture = "Unknown"
                        
                        # บันทึกค่าเก็บไว้ในความจำสำรอง
                        gesture_info = {"gesture": gesture, "conf": best_conf}
                    else:
                        # ข้ามการพยากรณ์! ดึงค่าเก่ารอบที่แล้วมาใช้เลย (ไม่กินสเปคเครื่อง)
                        gesture_info = cached_results[i]

                    current_frame_data.append(gesture_info)
                    
                    # ดึงค่ามาใช้วาดภาพ
                    gesture   = gesture_info["gesture"]
                    best_conf = gesture_info["conf"]

                    # ท่อนวาดภาพ (เหมือนเดิมเป๊ะ)
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                    )

                    x1, y1, x2, y2 = get_bounding_box(hand_landmarks, w, h)
                    box_color = (84, 125, 57) if best_conf >= CONF_THRESHOLD else (0, 165, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
                    label = f"{gesture}  {best_conf:.0%}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), box_color, -1)
                    cv2.putText(frame, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # อัปเดตหน่วยความจำรอบล่าสุด
                if frame_count % 5 == 0 or not cached_results:
                    cached_results = current_frame_data

            cv2.imshow("Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()