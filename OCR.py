import cv2
import time
import numpy as np 
import mxnet as mx
import os
#load pretrain model
model = mx.mod.Module.load('CNN_V1',2)
model.bind(for_training=False, data_shapes=[('data', (1,1,65,40))])

cap = cv2.VideoCapture("/home/lee/Desktop/project/machine (online-video-cutter.com).mp4")
Height = 195
Width = 240
count1 = 0
while(True):
    T = time.time()
    string = []
    substring = []
    ret, frame = cap.read()
    count1= count1+1
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 480, 720);
    cv2.imshow('frame', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret1,th1 = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
    crop_img = th1[1368:1610, 407:715]
    img = cv2.resize(crop_img, (240, 195), interpolation=cv2.INTER_NEAREST)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count =0 
    for i in range(0,Height,65):
        for j in range(0,Width,40):
            
            region = img[i:i+65,j:j+40]
            count = count+1
            im = np.array(region).reshape(1,1,65,40)
            im = mx.io.NDArrayIter(im)
            prob = model.predict(im)
            if np.argmax(prob.asnumpy()) == 10:
                
                if len(substring) != 5:
                    substring.append(" ")
                else:
                    substring.append(' ')
                    string.append(substring)
                    substring = []
            else:
                
                if len(substring) != 5:
                    substring.append(np.argmax(prob.asnumpy()))
                else:
                    substring.append(np.argmax(prob.asnumpy()))
                    string.append(substring)
                    substring = []
                    
    #print("frame : "+str(count1))
    st1 = ' '.join(str(st) for st in string[0])
    st2 = ' '.join(str(st) for st in string[2])
    print(st1[:6]+". "+st1[6:]) 
    print(st2[:6]+". "+st2[6:])
    print("Cost Time:%.2fs\n" % (time.time()-T))
    #os.system('clear')
cap.release()
cv2.destroyAllWindows()
