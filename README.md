# Priya's thesis project software

We use two different approaches:

1. Use Google's video intelligence API to track people in the video then use the
Levi and Hassner neural network to do gender classification on those frames.
(gcloudtest.py)

2. We use OpenCV's Harr Cascade Classifier to identify all faces in an image then
classify those faces using the Levi/Hassner algorithm. The idea behind this is
that we don't need to do object tracking specifically on a video as long as we
can keep a ratio of male/female faces in each frame to calculate the total on
screen time for male/female characters.
(vidtest.py)

To-Dos:

* vidtest.py needs to be tested with the caffe models from the gender/age
classification project

* We need to increase the accuracy of the face classifier and consider other
ways of classifying gender because as of now our algorithm only gives us the
gender ratios of the faces of people who are on screen and does not account for
when a person's face is not visible or is turned away.

* If we choose to go with the gcloud route then we need to add gender
classification to the the gcloud test script.

## How to run:

Using Python2 in the shell type:

This script runs a Haar cascade classifier to get the bounding boxes of the face
in each frame and uses the Levi Hassner model to get the gender label for each
face.
```
python vidtest.py -s [source video path]
```


This script simply returns the gcloud video intelligence results for a video.
With gcouldtest.py you must edit the *json_path* and *path* variables to be the path
to gcloud credentials json file and the path to the desired video to analyze
respectively.
```
python gcloudtest.py
```
