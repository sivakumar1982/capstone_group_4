First the videos in the videos folder has to be processed. Use the below cmd for the same
python gather_examples.py --input videos --output dataset --detector face_detector --skip 3

--input -> video folder under src dir
--output -> dataset folder under src dir
--detector - > contains a dnn caffe dnn model and its weights, that can extract face embeddings from a image. Because of this we see a
               perfect faces in the dataset folder
--skip -> how many frames to be skipped when converting videos to image dataset
Note : New folders under dataset will be created by the name of the videos in the video folder

Once the images embeddings are obtained, the same has to be trained.
python train_liveness.py --dataset dataset --model liveness.h5 --le le.pickle

This file uses a tensorflow neural net created under folder Liveliness

--dataset -> training data
--model -> Trained model is saved as .h5 file to be used in the LiveCam Detection.
--le -> will hold our lables as a disk file which is later loaded in the Livecam Detection

Once the Model is trained we can use it to run LiveCam Detection
python liveness_demo.py --model liveness.h5 --le le.pickle --detector face_detector

--model -> our previously trained and saved model
--le -> our label encoders
--detector -> to extract face embeddings from LiveCam Feed

