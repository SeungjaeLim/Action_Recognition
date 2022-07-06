# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
try:
#     %tensorflow_version 2.x
except Exception:
    pass

import os
import random
import re

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm.auto import tqdm

# DO NOT EDIT THE FOLLOWING LINES
# THESE LINES ARE FOR REPRODUCIBILITY
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

"""### 1. Load the UCF101 dataset

Using the UCF101 which is an action recognition dataset of realistic action videos, collected from YouTube, having 101 action categories. (*Soomro, K., Zamir, A. R., & Shah, M. (2012). UCF101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402.*)

![UCF101 Dataset](https://github.com/keai-kaist/CS470-Spring-2022-/blob/main/Lab3/May%2012/images/ucf101.jpg?raw=true)

The UCF101 dataset consists of 13,320 videos and their labels. Since your computing resource in Google Colab is somewhat limited, TA sampled half of the dataset, limited the length of videos to 64 frames, separated videos into frames and stored them to a single file in advance.
"""

import pickle

if not os.path.exists('ucf101.pickle'):
    !wget -O 'ucf101.pickle' 'https://www.dropbox.com/s/2558ailo46px55j/ucf101.pickle?dl=1'

    # If the link above is not working, you can also use the following link but it would be slower than the above.
    # !wget -O 'ucf101.pickle' 'http://cs492f.keai.io/ucf101.pickle'
    
with open('ucf101.pickle', 'rb') as input_file:
    dataset = pickle.load(input_file)
    
num_trains = len(dataset['train'])
num_validations = len(dataset['validation'])
num_tests = len(dataset['test'])

index_to_label = [
    'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 
    'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 
    'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 
    'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 
    'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 
    'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 
    'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 
    'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 
    'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 
    'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 
    'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 
    'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 
    'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 
    'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 
    'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 
    'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 
    'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 
    'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 
    'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 
    'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 
    'YoYo', 
]

"""Let's visualize what some of these videos and their corresponding labels look like."""

from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

def plot_frames(frames):
    figure = plt.figure(figsize=(frames.shape[1] / 72, frames.shape[2] / 72), dpi=72)
    image = plt.figimage(frames[0])
    
    def animate(step):
        image.set_array(frames[step])
        return (image, )
    
    video = FuncAnimation(
        figure, animate, 
        frames=len(frames), interval=33, 
        repeat_delay=1, repeat=True
    ).to_html5_video()
    
    display(HTML(video))

for frames, label in random.sample(dataset['train'], 3):
    plot_frames(tf.stack([tf.image.decode_jpeg(frame) for frame in frames]))
    print(frame[0] for frame in frames)
    print('Label:', index_to_label[label])

"""### 2. Preprocess the dataset

Unlike images and text, video data contains both spatial and temporal information. Therefore, to handle these data, you will use both convolutional neural networks and recurrent neural networks to recognize an action of the videos.

First, let's extract meaningful features from video frames using the pre-trained convolutional neural networks.

Define a model to extract meaningful features from the given video frame using the pre-trained convolutional neural networks of your choice.

This model should output a 1D vector for one given frame. That is,
- In: `(1, 256, 256, 3)` → Out: `(1, dimension of features of your choice)`
- In: `(5, 256, 256, 3)` → Out: `(5, dimension of features of your choice)`
- In: `(number of frames, 256, 256, 3)` → Out: `(number of frames, dimension of features of your choice)`
"""

# TODO: Define a model to extract features from the given video frame
#       using the pre-trained convolutional neural networks of your choice.

IMG_SIDE = 256
IMG_SHAPE = (256,256, 3)
cnn_model = tf.keras.Sequential([
  tf.keras.applications.InceptionV3(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet'),
  tf.keras.layers.GlobalAveragePooling2D()
])
cnn_model.trainable = False
cnn_model.summary()

""" **preprocess each frame** so it can be fed into the `cnn_model`"""

# This function extracts features from the given frames using the defined cnn_model
# - In: frames.shape = (number of frames, 256, 256, 3)
# - Out: features.shape = (number of frames, dimension of features of your choice)
def extract_features(frames, batch=32):
    # TODO: Preprocess each frame so it can be fed into the `cnn_model`

    frames = tf.keras.applications.inception_v3.preprocess_input(tf.cast(frames, tf.float32))

    
    features = tf.concat([
        cnn_model(frames[index:index + batch])
        for index in range(0, frames.shape[0], batch)
    ], axis=0)
    
    if features.shape[0] < max_length:
        features = tf.concat([
            features,
            tf.zeros((max_length - features.shape[0], *features.shape[1:]))
        ], axis=0)
    
    return features.numpy()

"""Now, you can extract features from the video but this task is very time-consuming. Therefore, `preprocess_dataset()` function which takes a dataset, extracts features from the dataset, stores the features into a file, and loads the features from the file."""

import os
import pickle

max_length = 32 # DO NOT CHANGE THIS NUMBER

def decode_frames(frames):
    return tf.stack([tf.image.decode_jpeg(frame) for frame in frames])

def preprocess_dataset(dataset, filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as input_file:
            return pickle.load(input_file)
    else:
        tensors = [
            (extract_features(decode_frames(frames[:max_length])), label)
            for frames, label in tqdm(dataset)
        ]
        
        X, y = zip(*tensors)
        X, y = np.array(X), np.array(y)
        
        with open(filename, 'wb') as output_file:
            pickle.dump((X, y), output_file, protocol=4) # protocol=4 supports a byte objects larget than 4GB
        
        return (X, y)

train_features, train_labels = preprocess_dataset(dataset['train'], f'train-dataset-{max_length}-{num_trains}.pickle')
validation_features, validation_labels = preprocess_dataset(dataset['validation'], f'validation-dataset-{max_length}-{num_validations}.pickle')
test_features, test_labels = preprocess_dataset(dataset['test'], f'test-dataset-{max_length}-{num_tests}.pickle')

"""Then, combine the features into batches."""

batch_size = 64

batch_train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).shuffle(512).batch(batch_size)
batch_validation_dataset = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels)).batch(batch_size)
batch_test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(batch_size)

"""### 3. Build the model
All videos are transformed into 2D tensors via convolutional neural networks. To process these tensors, let's build a recurrent neural network.


Define a recurrent neural network to recognize one of `num_classes` actions from the given video. Becaue all videos have different lengths, your `lstm_model` should take account this into account. To do that, TA added `tf.keras.layers.Masking` layer in advance.
"""

num_classes = len(index_to_label)

lstm_model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=0.0), # DO NOT REMOVE THIS LAYER

    # TODO: Define a recurrent neural network to recognize one of `num_classes` actions from the given video
    tf.keras.layers.LSTM(512, dropout=0.5, recurrent_dropout=0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(index_to_label), activation='softmax')
])

"""Then, compile your model with appropriate parameters."""

# TODO: Compile the model with appropriate parameters

lstm_model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer='rmsprop',  
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

"""### 4. Train the model
Now, train your `lstm_model` using train and validation dataset at least 10 epochs.
"""

batch_train_dataset

# TODO: Train the `lstm_model` using train and validation dataset at least 10 epochs
lstm_model.fit(
  batch_train_dataset,
  epochs = 25,

)

"""### 4. Evaluate accuracy
#### Problem 5
Let's evaluate the trained model using test dataset and print the test accuracy of the model. For your information, the accuracy of the model proposed by the authors who published the UCF101 dataset is 43.90%.
"""

# TODO: Evaluate the model using test dataset
test_loss, test_accuracy, k_accuracy = lstm_model.evaluate (batch_test_dataset)

print(f'Test accuracy: {test_accuracy:.4f}')
print(f'Test loss: {test_loss:.4f}')

"""Using the below cell, you can try to recognize an action of the test videos using your trined `lstm_model`."""

for frames, label in random.sample(dataset['test'], 3):
    print('Acutal:', index_to_label[label])
    
    print('Predicted:')
    features = extract_features(decode_frames(frames[:max_length]))
    predicted = lstm_model(tf.expand_dims(features, 0))[0]
    for confidence, index in zip(*tf.math.top_k(predicted, k=3)):
        print(f'- {index_to_label[index]} ({confidence.numpy():.4f})')
    
    plot_frames(decode_frames(frames))
    print()
