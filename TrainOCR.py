import mxnet as mx
import numpy as np
import os
import time
import logging
from matplotlib import pyplot as plt
from mxnet import gluon
import logging
logging.getLogger().setLevel(logging.DEBUG)
ctx = mx.gpu(0)

# creating a mxnet module.
data = mx.symbol.Variable('data')
conv1 = mx.symbol.Convolution(data=data, kernel=(7,7), num_filter=32)
relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
pool1 = mx.symbol.Pooling(data=relu1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=64)
relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
pool2 = mx.symbol.Pooling(data=relu2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
conv3 = mx.symbol.Convolution(data=pool2, kernel=(3,3), num_filter=64)
relu3 = mx.symbol.Activation(data=conv2, act_type="relu")
pool3 = mx.symbol.Pooling(data=relu2, pool_type="max",
                              kernel=(2,2), stride=(2,2))

flatten = mx.symbol.Flatten(data=pool3)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=32)
relu4 = mx.symbol.Activation(data=fc1, act_type="relu")
fc2 = mx.symbol.FullyConnected(data=relu4, num_hidden=11)
out = mx.symbol.SoftmaxOutput(fc2, name = 'softmax')


#load_data
T = time.time()
rec_path = os.path.expanduser('/home/lee/Desktop/machine02/model/ocrdata/')

batch_size = 200
train_data = mx.io.ImageRecordIter(
    path_imgrec = os.path.join(rec_path, 'train_data/ocrdata_train.rec'),
    path_imgidx = os.path.join(rec_path, 'train_data/ocrdata_train.idx'),
    data_shape  = (1,65,40),
    batch_size  = batch_size,
    shuffle     = True
)
val_data = mx.io.ImageRecordIter(
    path_imgrec = os.path.join(rec_path, 'val_data/ocrdata_val.rec'),
    path_imgidx = os.path.join(rec_path, 'val_data/ocrdata_val.idx'),
    data_shape  = (1,65,40),
    batch_size  = batch_size,
    shuffle     = True
)
print("Load Image Time:"+str(time.time()-T))

batch = train_data.next()
data = batch.data[0]
label = batch.label[0]

for i in range(0,6):
    plt.subplot(1, 6, i+1)
    plt.title(label[i])
    plt.imshow(data[i].asnumpy().transpose((1,2,0)).reshape(65,40),cmap='gray')
plt.show()


logging.getLogger().setLevel(logging.DEBUG)
progress_bar = mx.callback.ProgressBar(total=2)
model = mx.mod.Module(symbol=out, context=ctx)
model.fit(train_data,
        eval_data=val_data,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01},
        eval_metric='acc',
        batch_end_callback = mx.callback.Speedometer(batch_size, 200),
        num_epoch=5)

num_epoch = 2
model.save_checkpoint('CNN_V1', num_epoch, save_optimizer_states=True)

"""
import mxnet as mx
model = mx.mod.Module.load('CNN_V1',2)
model.bind(for_training=False, data_shapes=[('data', (1,1,65,40))])

import cv2
import time
import numpy as np 
import mxnet as mx
"""

cap = cv2.VideoCapture("/home/lee/Desktop/machine.mp4")
Height = 195
Width = 240
count1 = 0

while(True):
    string = []
    substring = []
    ret, frame = cap.read()
    count1= count1+1
    #frame=np.rot90(frame,3)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 480, 720);
    cv2.imshow('frame', frame)
    cv2.imshow('frame', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret1,th1 = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
    crop_img = th1[1368:1610, 407:715]
    img = cv2.resize(crop_img, (240, 195), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('frame1', img)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

    count =0 
    for i in range(0,Height,65):
        for j in range(0,Width,40):
            
            region = img[i:i+65,j:j+40]
            count = count+1
            im = np.array(region).reshape(1,1,65,40)
            #print(im.shape)
            im = mx.io.NDArrayIter(im)
            prob = model.predict(im)
            #print(np.argmax(prob.asnumpy()))
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
    print("frame : "+str(count1))
    st1 = ' '.join(str(st) for st in string[0])
    st2 = ' '.join(str(st) for st in string[2])
    print(st1[:6]+". "+st1[6:]) 
    print(st2[:6]+". "+st2[6:])
cap.release()
cv2.destroyAllWindows()
