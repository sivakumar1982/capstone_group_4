import ffmpeg, pprint, subprocess
import cv2
import glob
import argparse
import os
import shutil


ap = argparse.ArgumentParser()
ap.add_argument("-vd", "--video_dataset", required=True, help="path to input directory of faces + images")
ap.add_argument("-o", "--output", required=True, help="path to output directory of faces + images")
args = vars(ap.parse_args())


def check_rotation(path_video_file):
    print("Check Rotation  File {}".format(path_video_file))
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


def process_videos(args):
    print("[INFO] Processing Videos...")
    files = glob.glob(args['video_dataset'] + '/*.*')
    print(files)
    for file in files:
        name = file.split('\\')
        name1 = name[-1]
        name = name1.split(".")[0]
        print('\n' + name)
        print('\n' + "[INFO] Processing Video File " + file + '\n')
        vidcap = cv2.VideoCapture(file)
        try:
            rotateCode = None
            rotateCode = check_rotation(file)
        except Exception as e:
            print(e)
        success, image = vidcap.read()
        count = 0
        BASE_DIR = r'E:\AIML-GL\Capstone\src\dataset'
        CHILD_DIR = os.path.join(BASE_DIR, name)
        _dir = os.path.isdir(CHILD_DIR)
        if _dir == False:
            os.mkdir(CHILD_DIR)
            os.chdir(CHILD_DIR)
        else:
            os.chdir(CHILD_DIR)
        while success:
            if rotateCode is not None:
                image = correct_rotation(image, rotateCode)
                cv2.imwrite("_%d.jpg" % count, image)  # save frame as JPEG file
                success, image = vidcap.read()
                print('\r', 'Read a new frame: {}'.format(success), end='')
                count += 20  # i.e. at 30 fps, this advances one second
                vidcap.set(1, count)
            else:
                cv2.imwrite("_%d.jpg" % count, image)  # save frame as JPEG file
                success, image = vidcap.read()
                print('\r', 'Read a new frame: {}'.format(success), end='')
                count += 20  # i.e. at 30 fps, this advances one second
                vidcap.set(1, count)
        vidcap.release()
        cv2.destroyAllWindows()
        DES_FILE = r'E:\AIML-GL\Capstone\src\dataset_video\Processed_Videos'
        shutil.move(file, DES_FILE+ "\\"+ name1)
        #os.remove(file)




if __name__ == "__main__":
    process_videos(args)
