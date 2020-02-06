# Importing Necessary Libraries
import numpy as np
import cv2
import os
# import json
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import load_model


# Defining Predictor Class
class Classifier:
    def __init__(self,base_model,class_model,class_list):
        if base_model=="VGG16":
            self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=(160,120,3))
        elif base_model=="ResNet":
            self.base_model =  ResNet50(weights='imagenet', include_top=False, input_shape=(160,120,3))
        else:
            print("Error: Base Model Not Defied")
        self.class_model=load_model("F:/Projects/Computer-Vision-with-Python/My Projects/UAV/Threat Level Classifier/Models/"+class_model)
        self.class_list=class_list
        print("Classifier Online")
        
    def classify(self,vid_name,save_name):
        print("Starting Video Capture")
        cap = cv2.VideoCapture(vid_name)
        
#        Checking Empty Video
        
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if(frameCount==0):
            print("Empty Video")
            return [-1]
        
        X = np.empty((int(frameCount/2), 160, 120, 3), np.dtype('uint8'))
        fc = 0
        ret = True
        
#       Frame Extraction Initiated
        count=0
        while (count < 2*int(frameCount/2)  and ret):
            ret, frame = cap.read()
            if(count%2==0):
                count+=1
                continue
            frame=cv2.resize(frame,(120,160))
            X[fc]=frame
            fc += 1
            count+=1
            # cv2.imshow("Frame",frame)
            # cv2.waitKey(20)
        cap.release()
        # cv2.destroyAllWindows()
        
#        Frame Extraction Complete, Starting Prediction 
        x_features = self.base_model.predict(X)
        x_features = x_features.reshape(x_features.shape[0],
                     x_features.shape[1] * x_features.shape[2],
                     x_features.shape[3])
        pred = self.class_model.predict(x_features)
        
#        Display Output
        cap1=cv2.VideoCapture(vid_name)
        frameCount = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_H=int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_W=int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        out = cv2.VideoWriter(save_name,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_W,frame_H))
        print("Got Pred")
        print(frameCount)
        count = 0
        fc = 0
        ret = True
        while (fc < int(frameCount/2)  and ret):
            ret, frame = cap1.read()
            if(count%2==0):
                count+=1
                continue
            if(ret!=True):
                break
            text="Anomaly "
            prediction=pred[fc][0]
            if(pred[fc][1]-pred[fc][0]>0):
                text="Normal "
                prediction=pred[fc][1]-pred[fc][0]
            frame=cv2.putText(frame,text+str(prediction),(0,int(frame_H/5)),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
            out.write(frame)
            cv2.imshow("Frame",frame)
            c=cv2.waitKey(20)
            if c==27:
                break
            fc += 1
            count += 1
        cap.release()
        cv2.destroyAllWindows()
        out.release()
        print("File_Saved")
        
#        Returning Output
        return pred    

# Setting Up Model
Model=Classifier("VGG16",'Alt_4_ResNet.h5',['Anomaly','Normal'])

i=0
mem_loc = "F:/Projects/SIH/Anomaly-Videos-Part-1/Abuse"
vids = os.listdir(mem_loc)
vid_name= vids[i]
save_loc = "F:/Projects/SIH/Anomaly_Video/"
while True:
    # Predicting on Video
    text="00"
    if i>9:
        text="0"
    elif i>99:
        text=""
    print(mem_loc +"/"+ vid_name + '.mp4')
    pred=Model.classify(mem_loc +"/"+ vid_name,save_loc+vid_name.split(".")[0]+".avi")
    i+=1
    vid_name = vids[i%len(vids)]
    if cv2.waitKey(1)==27:
        break
    if(len(pred)==1):
        break
    
# Converting Prediction Data to suitable format 
    # l=len(pred)
    # num_partitions=2
    # Arr1=[]
    # Arr2=[]
    # for i in range(num_partitions):
    #     Arr1.append(str(np.average(pred[i*int(l/num_partitions):(i+1)*int(l/num_partitions),0])))
    #     Arr2.append(str(np.average(pred[i*int(l/num_partitions):(i+1)*int(l/num_partitions),1])))
    # data={"Anomaly":Arr1,"Normal":Arr2,"Time_Taken":[int((end-start)/num_partitions)]*num_partitions}
    
    # # Saving Data in File
    # File_loc="Data"
    # with open(File_loc+".json","w") as f:
    #     json.dump(data,f)
