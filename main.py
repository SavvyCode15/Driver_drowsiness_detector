import dlib 
import cv2
import imutils
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer

mixer.init()
mixer.music.load("Detected.wav")
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh=0.20
flag=0
frame_check=20
(lStart, lEnd)= face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd)= face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']


detect=dlib.get_frontal_face_detector()
predict=dlib.shape_predictor("Face_Landmarks_Model.dat")
cap =cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    frame=imutils.resize(frame, width=450)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    subjects=detect(gray, 0)
    for subject in subjects:
        shape=predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        lefteye =  shape[lStart:lEnd]
        righteye =  shape[rStart:rEnd]
        leaftEar = eye_aspect_ratio(lefteye)
        rightEar = eye_aspect_ratio(righteye)
        ear=(leaftEar + rightEar)/2.0
        leftEyeHull = cv2.convexHull(lefteye)   
        rightEyeHull = cv2.convexHull(righteye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear<thresh:
            flag+=1
            print(flag)
            if flag>=frame_check:
                cv2.putText(frame, "******ALERT******", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(frame, "******ALERT******", (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                mixer.music.play()
        else: 
            flag=0        
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()            