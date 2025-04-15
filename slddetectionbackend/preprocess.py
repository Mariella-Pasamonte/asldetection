def pad_to_84(features):
    return features + [0]*42

def predictASL(model, landmarks):
    if len(landmarks) == 42:
        return model.predict([pad_to_84(landmarks)]) #kuwangan if 42 ra, mao use pad_to_84
        
    if len(landmarks) == 84: 
        return model.predict([landmarks]) #E predict dayon if 84 na daan
    return None

                