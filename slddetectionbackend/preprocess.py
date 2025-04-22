
def predictASL(model63, model126, landmarks):
    if len(landmarks) == 63:
        return model63.predict([landmarks]) #kuwangan if 42 ra, mao use pad_to_84
        
    if len(landmarks) == 126: 
        return model126.predict([landmarks]) #E predict dayon if 84 na daan
    return None

                