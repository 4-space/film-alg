import cv2
import numpy as np
import argparse
import caffe

"""
This is a test for using opencv to segment a video into individual frames then
using pre-trained NN models to classify faces in those frames to calculate
a on-screen gender ratio for the video!
"""

ap = argparse.ArgumentParser()
ap.add_argument('-s','--source', required=True, help="Path to video file")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["source"])
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

gender_net_pretrained='./gender_net.caffemodel'
gender_net_model_file='./deploy_gender.prototxt'

mean_filename='./mean.binaryproto'
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]

gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

gender_list=['Male','Female']

start = False
frames = 3
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        prediction = gender_net.predict([roi_color])
        label = gender_list[prediction[0].argmax()]
        putText(frame, label, Point(x, y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0), 2.0);

    cv2.imshow('lizzo test', frame)

    if frames > 0:
        frames = frames - 1
    else:
        if start == False:
            raw_input("Push to start.")
            start = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
