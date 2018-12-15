import numpy as np
import cv2
from collections import deque
from mnist import MNIST
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import os

cnn_model = load_model('emnist_cnn_model.h5')
mlp_model = load_model('emnist_mlp_model.h5')

letter_mapping = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}

# define the upper and lower boundaries for a color to be considered 'Blue'
blueLower = np.array([100,60,60])
blueUpper = np.array([140,255,255])

# define a 5x5 kernel for erosion and dialation
kernel = np.ones((5,5),np.uint8)

# define black borad
blackboard = np.zeros((480,640,3),dtype=np.uint8)
alphabet = np.zeros((200,200,3),dtype=np.uint8)

# setup deques to store alphabet drawn on screen
points = deque(maxlen=512)

# define prediction variables
pred1 = 26
pred2 = 26

index = 0

# load video
camera = cv2.VideoCapture(0)

# keep looping
while True:
    # grab the current paint window
    (grabbed,frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # determine which pixels fall within the blue boundaries and then blur the image
    blueMask = cv2.inRange(hsv,blueLower,blueUpper)
    blueMask = cv2.erode(blueMask,kernel,iterations=2)
    blueMask = cv2.morphologyEx(blueMask,cv2.MORPH_OPEN,kernel)
    blueMask = cv2.dilate(blueMask,kernel,iterations=1)
    
    # find contours (bottle cap in this case) in the image
    (_,cnts,_) = cv2.findContours(blueMask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None
    
    # check to see if any contours were found
    if len(cnts)>0:
        # sort the contours and find the largest one, assuming this contour correspondes to the area of bottle cap
        cnt = sorted(cnts,key=cv2.contourArea,reverse=True)[0]
        # get the radius of the enclosing circle around the found contour
        ((x,y),radius) = cv2.minEnclosingCircle(cnt)
        # draw the circle around the contour
        cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
        # get moments to calculate the center of the contour
        M = cv2.moments(cnt)
        center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        points.appendleft(center)
        
    elif len(cnts)==0:
        if len(points) != 0:
            blackboard_gray = cv2.cvtColor(blackboard,cv2.COLOR_BGR2GRAY)
            blur1 = cv2.medianBlur(blackboard_gray,15)
            blur1 = cv2.GaussianBlur(blur1,(5,5),0)
            thresh1 = cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            blackboard_cnts = cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
            if len(blackboard_cnts)>=1:
                cnt = sorted(blackboard_cnts,key=cv2.contourArea,reverse=True)[0]
                
                if cv2.contourArea(cnt)>1000:
                    x,y,w,h = cv2.boundingRect(cnt)
                    alphabet = blackboard_gray[y-10:y+h+10,x-10:x+w+10]
                    newImage = cv2.resize(alphabet,(28,28))
                    newImage = np.array(newImage)
                    newImage = newImage.astype('float32')/255
                    
                    pred1 = mlp_model.predict(newImage.reshape(1,28,28))[0]
                    pred1 = np.argmax(pred1)
                    
                    pred2 = cnn_model.predict(newImage.reshape(1,28,28,1))[0]
                    pred2 = np.argmax(pred2)
            # empty the points deque and blackboard
            points = deque(maxlen=512)
            blackboard = np.zeros((480,640,3),dtype=np.uint8)
    # connect the points with a line
    for i in range(1,len(points)):
        if points[i-1] is None or points[i] is None:
            continue
        cv2.line(frame,points[i-1],points[i],(0,0,0),2)
        cv2.line(blackboard,points[i-1],points[i],(255,255,255),8)
    # put the results on the screen
    cv2.putText(frame,'Multilayer perceptron: '+str(letter_mapping[int(pred1)+1]),
               (10,410),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.putText(frame,'Convolution Neural Network: '+str(letter_mapping[int(pred2)+1]),
               (10,440),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    
    # show the frame
    cv2.imshow('Alphabets Recognition in Real Time',frame)
    
    # stop the loop util pressing the'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clean up the camera and close any open window
camera.release()
cv2.destroyAllWindows()