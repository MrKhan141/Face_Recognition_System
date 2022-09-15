#Write a python script to captures the images from your webcam video stream
#Extract all faces from the image frame(using haarcascades)
#store the faces information in numpy array...
import cv2
import numpy as np

#Init the object
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip = 0 ;
face_data = []
dataset_path = "./Data/"
face_section=0
filename=input("Enter the filename:")
while True :
      ret,frame = cap.read()
      gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      if ret == False :
      	continue
      
      faces = face_cascade.detectMultiScale(frame,1.3,5)
      faces = sorted(faces,key=lambda f:f[2]*f[3])
     
      for (x,y,w,h) in faces[-1: ] :
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) 

            #Extract (crap out the face from image):Region of interest
            offset = 10
            face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
            face_section = cv2.resize(face_section,(200,200))

            skip += 1
            if skip%10==0:
            	face_data.append(face_section)
            	print(len(face_data))

      cv2.imshow("Video Frame",frame)
      cv2.imshow("Face section",face_section)
      #wait for user input -q , to exit

      key_pressed = cv2.waitKey(1) & 0xFF
      if key_pressed == ord('q') : 
      	break
#convert face list array into numpy array...
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save the file in directory..
np.save(dataset_path+filename+'.npy',face_data)
print("Data successfully saved at"+dataset_path+filename+'.npy')

cap.release()
cv2.destroyAllWindows()