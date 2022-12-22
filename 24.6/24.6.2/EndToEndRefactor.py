# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:47:54 2022

@author: darth
"""

#%% Setup
import warnings
#warnings.filterwarnings("ignore")

import joblib
from pydub import AudioSegment
import librosa
import numpy as np
import pandas as pd
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} # Turn off uneccsary warnings

import tensorflow as tf
import argparse

#%% Setup CMD argument parsing

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path to MP3")
args = parser.parse_args()

#%% Import Pipeline Objects and Models
LabelEncode = joblib.load("LabelEncoder.jblb")
TFEncode = joblib.load("TFEncoder.jblb")
TFScale = joblib.load("TFScaler.jblb")
XBGModel = joblib.load("XGBTuned.jblb")
TFModel = tf.keras.models.load_model('DL.h5')

#%% Create Model Object
class ModelAssets:

    #%% Data Feature Pipeline
    def feature_extraction(self,filename, accent = None, sampling_rate=48000):
        #path = 'cv-valid-'+folder+'/'+filename
        features = list()
        print(1)
        sound = AudioSegment.from_file(filename) # Load Audio File
        audio = np.array(sound.get_array_of_samples(),dtype=np.float32)
        #Calculate Features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate))
        features.append(accent)
        features.append(spectral_centroid)
        features.append(spectral_bandwidth)
        features.append(spectral_rolloff)
        mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate,n_mfcc = 12, lifter = 5)
        #Convert MFCC's to mean and standard deviation to reduce dimensions
        for el in mfcc:
            features.append(np.mean(el))
            features.append(np.std(el))
        dataset = []
        dataset.append(features)
        #Setup Feature dataframe
        S_df = pd.DataFrame(dataset,columns = ["label", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
                                            "mfcc1", "Std1",
                                            "mfcc2", "Std2",
                                            "mfcc3", "Std3",
                                            "mfcc4", "Std4",
                                            "mfcc5", "Std5",
                                            "mfcc6", "Std6",
                                            "mfcc7", "Std7",
                                            "mfcc8", "Std8",
                                            "mfcc9", "Std9",
                                            "mfcc10", "Std10",
                                            "mfcc11", "Std11",
                                            "mfcc12", "Std12",
                                            ])
        S_df.drop('label',axis = 1,inplace = True) # Drop the label column, not needed for predictions
        print(2)
        return S_df
    
    #%% Predict Using XGBoost
    def predictWithXGB(self):
        print(11)
        prediction = XBGModel.predict(self.f.values)
        print(11)
        return str(LabelEncode.classes_[prediction])
        
    #%% Predict using DL
    def predictWithDL(self):
        prediction = TFModel.predict(TFScale.transform(self.f.values)).argmax(1)
        return str(LabelEncode.classes_[prediction])
    
    def createFeatures(self, Filename):
        self.f = self.feature_extraction(Filename)