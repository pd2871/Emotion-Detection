# Emotion Detection using CNN
### Aim 
- To detect the face from live camera frame and use CNN to classify the facial expression of person in the frame(Happy, Angry, Sad, Surprised, Calm, Neutral)


### Description
- This project is based on CNN and face recognition technique using HAAR CASCADE.
- Accuracy of the model is around 55% since facial expressions seems to be similar(like calm and neutral are similiar, angry and sad seems similar)
- Face Detection process is fast using HAAR CASCADE but however it can be improved using MTCNN
- Retraining with different models will be taking a lot of time since the images are around 37000 with 150*150 pixels, so its beter to use the pretrained model(took me 4 hours for   20 epochs). 
- But the images size can be decreased to 50*50 for faster training

### Process
- Used CNN to classify the input images into emotions like Happy, Sad, Angry,etc. with accuracy of around 55%. Saved the model 
- Used OpenCV to detect face and extract the face from live frames
- Applied the saved model to the detected faces 
- Model predicted the emotions of the detected face
- Used OpenCV to show the frame along with the prediciton made by model and the bounding box detected by the HAAR CASCADE


### Frameworks
- Tensorflow
- Keras
- Scikit-learn
- OpenCV

### Libraries
- tqdm
- Numpy
- Matplotlib
