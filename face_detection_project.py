#Recognize the face using algorithm like KNN , Logistic regression,SVM etc.
#1.Load the training data(numpy array of all persons) 
#  x-values are stored in numpy array
#  y-values are assign for each person
#2.read a video stream using opencv.
#3.Extract the images/faces from it.
#4.Use KNN to predict faces .
#5.Map the predicted id to the name of the user.
#6.Display the prediction on the screen.
import cv2
import numpy as np
import os

########## KNN CODE ############
def distance(v1, v2):
      # Eucledian 
      return np.sqrt((v1[0]-v2[0])**2)+((v1[1]-v2[1])**2) 

def knn(train, test, k=10):
      dist = []
      
      for i in range(train.shape[0]):
            # Get the vector and label
            ix = train[i, :-1]
            iy = train[i, -1]
            # Compute the distance from test point
            d = distance(test, ix)
             
            dist.append([d, iy])
      # Sort based on distance and get top k
      dk = sorted(dist, key=lambda x: x[0])[:k]
      # Retrieve only the labels
      labels = np.array(dk)[:, -1]
      
      # Get frequencies of each label
      output = np.unique(labels, return_counts=True)
      # Find max frequency and corresponding label
      index = np.argmax(output[1])
      return output[0][index]
################################

#init camera
cap = cv2.VideoCapture(0)
#Face detection
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0 
face_data=[]
dataset_path="./Data/"
labels = []
class_id = 0
names={} #Mapping between class_id and names.

#Data Preparation...

for fx in os.listdir(dataset_path):
      if fx.endswith('.npy'):
            names[class_id]=fx[:-4] #Mapping of id - name.
            print("Loaded"+fx)
            data_item = np.load(dataset_path+fx)
            face_data.append(data_item)
      #Creat labels for the class...
      target = class_id * np.ones((data_item.shape[0],))
      class_id += 1
      labels.append(target)
face_dataset = np.concatenate(face_data,axis=0)
face_label = np.concatenate(labels,axis=0).reshape((-1,1))

trainset = np.concatenate((face_dataset,face_label),axis=1)
 

#Testing 
while True :
      ret,frame=cap.read()
      gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      if ret == False :
            continue
      faces = face_cascade.detectMultiScale(frame,1.3,10)

      for face in faces :
            x,y,w,h = face 

            #Get the face ROI(Region of Interest)
            offset = 10
            face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
            face_section = cv2.resize(face_section,(200,200))
            #Predicted Label
            out = knn(trainset,face_section.flatten()) 
            #Display the name and box on face..
            Pred_name=names[int(out)]
            cv2.putText(frame,Pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

      cv2.imshow("Faces",frame)
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


















