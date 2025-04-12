import cv2
import tensorflow
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

def detect_hand_landmarks2D(img):
    results = hands.process(img)
    landmarks_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks_list.append(landmark.x)  # x coordinate
                landmarks_list.append(landmark.y)  # y coordinate
    return landmarks_list

def pad_to_84(features):
    return features + [0]*42

def predictASL(model, landmarks):
    if len(landmarks) == 42:
        return model.predict([pad_to_84(landmarks)]) #kuwangan if 42 ra, mao use pad_to_84
        
    if len(landmarks) == 84: 
        return model.predict([landmarks]) #E predict dayon if 84 na daan
    return None

                