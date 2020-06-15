# USAGE
# python gather_examples.py --input videos/real.mov --output dataset/real --detector face_detector --skip 1
# python gather_examples.py --input videos/fake.mp4 --output dataset/fake --detector face_detector --skip 4

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
from glob import glob
from pathlib import Path
import ffmpeg

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16,
                help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


def check_rotation(path_video_file):
    print("Check Rotation File : {}".format(path_video_file))
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)
    rotCode = meta_dict['streams'][0]['tags']['rotate']
    rotCode = int(rotCode)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if rotCode == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif rotCode == 180:
        rotateCode = cv2.ROTATE_180
    elif rotCode == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
    return rotateCode


def correct_rotation(frame, rotateCode):
    return cv2.rotate(frame, rotateCode)


# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
# files = glob(args['input'] + '/*.*', recursive=True)

path = Path(args['input'])
files = path.rglob('*.*')
for file in files:
    file = str(file)
    name = file.split('\\')
    name1 = name[-1]
    name = name1.split(".")[0]
    if 'fake' in name.lower():
        name = 'Fake'
    print(name)
    BASE_DIR = r'C:\anton\py\capstone1\src\dataset'
    CHILD_DIR = os.path.join(BASE_DIR, name)
    print(CHILD_DIR)
    _dir = os.path.isdir(CHILD_DIR)
    if _dir == False:
        os.mkdir(CHILD_DIR)
    else:
        pass
    vs = cv2.VideoCapture(file)
    try:
        rotateCode = None
        rotateCode = check_rotation(file)
    except Exception as e:
        print(e)
    read = 0
    saved = 0

    # loop over frames from the video file stream
    while True:
        # grab the frame from the file
        (grabbed, frame) = vs.read()
        if rotateCode is not None:
            frame = correct_rotation(frame, rotateCode)

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # increment the total number of frames read thus far
        read += 1

        # check to see if we should process this frame
        if read % args["skip"] != 0:
            continue

        # grab the frame dimensions and construct a blob from the frame
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]

            # write the frame to disk
            p = os.path.sep.join([CHILD_DIR, "{}.png".format(saved)])
            cv2.imwrite(p, face)
            saved += 1
            print("[INFO] saved {} to disk".format(p))

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
