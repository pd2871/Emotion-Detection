import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from tensorflow.keras.models import load_model
classifier = load_model('model1.h5')

cap = cv2.VideoCapture(0)
haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    ret, frame = cap.read()
    if ret:
        fcs=[]
        bbx=[]
        preds=[]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray)
        for face in faces:
            x,y,w,h = face
            x2 = x+w
            y2 = y+h
            fc = gray[y:y2, x:x2]
            fc = cv2.resize(fc, (150, 150))
            fc = np.array(fc, dtype='float32')
            fc = np.reshape(fc, (150, 150, 1)) #reshaping from (1,150,150) to (150,150,1)
            fc = np.expand_dims(fc, axis=0)  # Changing to 4d for CNN
            fcs.append(fc)
            bbx.append((x,y,x2,y2))

        if(len(fcs))>0:
            preds = classifier.predict(fcs, batch_size=32)

        for (box,pred) in zip(bbx,preds):
            (x,y,x2,y2) = box
            prediction = np.argmax(pred)
            if prediction == 0:
                emotion = "Angry"
                color = (0, 0, 255)
            elif prediction == 1:
                emotion = "Disgusted"
                color = (0, 255, 255)
            elif prediction == 2:
                emotion = "Fearful"
                color = (255, 25, 25)
            elif prediction == 3:
                emotion = "Happy"
                color = (0, 255, 0)
            elif prediction == 4:
                emotion = "Neutral"
                color = (100, 255, 10)
            elif prediction == 5:
                emotion = "Sad"
                color = (100, 50, 150)
            else:
                emotion="Surprised"
                color = (250, 255, 0)

            cv2.rectangle(frame, (x, y), (x2, y2), color, 3)  # Putting rectangle of bbox in frames
            cv2.rectangle(frame, (x, y - 40), (x2, y), color, -1)

            cv2.putText(frame, emotion, (x+100, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Putting text in live frames
            cv2.imshow('frame', frame)

        if (cv2.waitKey(20) == ord('q')) or (cv2.waitKey(20) == 27):
            # pressing 'q' or 'esc' keys destroys the window
            break

        else:
            print("No faces detected")


    else:
        print("No frames detected")
        break

cap.release()
cv2.destroyAllWindows()

"""
The accuracy of the classifier can be increased by changing it with a new, more accurate classifier
or fine tuning the current one again
"""