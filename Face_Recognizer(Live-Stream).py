# Importing Necessary Libraries
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from imutils import face_utils
import json
import os
from fr_utils import *
from inception_blocks_v2 import *
import dlib
import requests

# Initializing Face Detection & Recognition Models
FR_model = load_model('nn4.small2.v1.h5')
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
threshold = 0.075

# Initializing Face Database
face_database = {}
id_data = pd.read_csv("Database.csv")
for name in os.listdir('images'):
	for image in os.listdir(os.path.join('images',name)):
		identity = os.path.splitext(os.path.basename(image))[0]
		face_database[identity] = fr_utils.img_path_to_encoding(os.path.join('images',name,image), FR_model)


# Initializing Video writers
frame_H=480
frame_W=640
out = cv2.VideoWriter("./Face_Rec_Video/Vid_0.avi" ,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_W,frame_H))
anomaly = cv2.VideoWriter("./Original_Vid/Vid_0.avi" ,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_W,frame_H))

# Fetching and Predicting on Live_Stream
links = []
names = []
livestream_url = "http://192.168.139.43:8080/shot.jpg" #edit according to server
json_url = "http://192.168.150.46:3001/api/data" #edit according to server
i=0
while True:
# Fetching Frame from live stream
    img_resp = requests.get(livestream_url)
    img_arr = np.array(bytearray(img_resp.content),dtype = np.int8)
    frame = cv2.imdecode(img_arr,-1)
    frame = cv2.resize(frame,(640,480))
    link = []
    name_data=[]
    frame = cv2.flip(frame, 1) # Correcting for left-right flip while recording
    anomaly.write(frame)
# Face Recognition
    faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    for face in faces:
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi = frame[y:y+h, x:x+w]
        encoding = img_to_encoding(roi, FR_model)
        if(type(encoding[0])==int):
            continue
        min_dist = 100
        identity = None
        for(name, encoded_image_name) in face_database.items():
            dist = np.linalg.norm(encoding - encoded_image_name)
            if(dist < min_dist):
                min_dist = dist
                identity = name
            # print('Min dist: ',min_dist)
        if min_dist < threshold:
            cv2.putText(frame, "Face : " + identity[:-1].split("_")[0], (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            name_data.append(identity[:-1].split("_")[0])
            link.append(id_data.loc[id_data["Names"]==identity[:-1].split("_")[0]]["Link"].array)
        else:
            cv2.putText(frame, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            link.append("-1")
            name_data.append("-1")
    cv2.imshow('Face Recognition System', frame)
    if(cv2.waitKey(1) == 27):
        break
    out.write(frame)
    names.append(name_data)
    links.append(link)
    if(i%100==0):
	anomaly.release()
	anomaly = cv2.VideoWriter("./Original_Vid/Vid_"+str(int(i/100))+".avi" ,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_W,frame_H)) 
        out.release()
        out = cv2.VideoWriter("./Face_Rec_Video/Vid_"+str(int(i/100))+".avi" ,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_W,frame_H))
# Sending Data for Analytics
    if(i%10 == 0):
        df = pd.DataFrame({"Names":names,"Links":links})
        print(df)
        resp = requests.post(json_url, data = df.to_json())
        names=[]
        links=[]
        print(resp.text)
    i+=1
df = pd.DataFrame({"Names":names,"Links":links})
resp = requests.post(json_url, data = df.to_json())
print(resp.text)
cv2.destroyAllWindows()
out.release()
