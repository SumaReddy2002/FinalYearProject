import cv2
import winsound
import imutils
from imutils.video import VideoStream

# Load pre-trained model from disk
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# Specify the class labels MobileNet SSD was trained to detect
CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
           'sofa', 'train', 'tvmonitor']

# initialize the video streams and allow them to warmup
print("[INFO] starting video streams...")
vs1 = VideoStream(src=0).start()
vs2 = VideoStream(src=1).start()

# loop over the frames from the video streams
while True:
    # grab the frames from the threaded video streams
    frame1 = vs1.read()
    frame2 = vs2.read()
    
    # resize the frames to have a width of 400 pixels
    frame1 = imutils.resize(frame1, width=400)
    frame2 = imutils.resize(frame2, width=400)
    
    # pass the blob through the network and obtain the detections and predictions
    blob1 = cv2.dnn.blobFromImage(cv2.resize(frame1, (300, 300)), 0.007843, (300, 300), 127.5)
    blob2 = cv2.dnn.blobFromImage(cv2.resize(frame2, (300, 300)), 0.007843, (300, 300), 127.5)
    
    net.setInput(blob1)
    detections1 = net.forward()
    
    net.setInput(blob2)
    detections2 = net.forward()
    
    person1_present = any(detections1[0, 0, i, 1] == 15 for i in range(detections1.shape[2]))
    person2_present = any(detections2[0, 0, i, 1] == 15 for i in range(detections2.shape[2]))

    # if person is not present in either of the frames, play a beep sound
    if not person1_present or not person2_present:
        winsound.Beep(1000, 1000) 

