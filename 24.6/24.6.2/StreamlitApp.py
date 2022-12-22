# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:34:33 2022

@author: darth
"""

import streamlit as st
import streamlit.components.v1 as components
from EndToEndRefactor import ModelAssets
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import io
import pandas as pd
#%%Intro
st.title("Audio Accent Identifier")
st.header("Understanding Audio")
st.write("Raw audio files consist of an analog signal, in the PCM format. Pulse-code modulation (PCM) is a method used to digitally represent sampled analog signals. It will look like the following sample.")

image = Image.open('ExampleWAV.PNG')
st.image(image, caption='Example Audio File')

st.write('This audio format contains all the data needed to replay the recorded sound. In our case we will be using the CommonVoice dataset from Kaggle, a repository of voice recordings along with some user input about the person making the recording. This includes their country of origin. Using the audio sample with the country we will try to make an accent recogniser.')
#%% The Data
st.subheader("Breaking Down the Data")
st.write("In order to train any sort of algorithm it is necessary to first look at the data and understand what the set holds.")

with st.expander("Audio Country Counts"):
    image = Image.open('AccentCounts.png')
    st.image(image, caption='Audio Country Counts')

st.write("The above graph of the accent counts shows that there are a large number of accents and over 60,000 total samples in the training set. This would be ok except that many of the countries have very few samples. This will require us to rework the data to help with the imbalance and number of classes.")
with st.expander("Reduced Countries"):
    image = Image.open('ReducedAccents.png')
    st.image(image, caption='Reduced Countries')

st.write("This chart shows the reduction of the number of classes or countries by grouping many of the ones with low samples into an \'other\' category.")

st.subheader("Machine Readable Features")
st.write("A classic way to interpret audio data is by using Mel Frequency Cepstral Coefficients (MFCC's). Below is the wikipedia page, its intro give a clear and concise description of what the are and how to use them.")

with st.expander("MFCC Wikipedia"):
    components.iframe("https://en.wikipedia.org/wiki/Mel-frequency_cepstrum",scrolling=True,height =450)
image = Image.open('MFCCExample.jpeg')
st.image(image, caption='Example of MFCC Spectrogram')

st.write("There are a few other features that we will be using: spectral centroid, spectral bandwidth and spectral rolloff.")
st.markdown("- " + 'Spectral Centroid is a measure of the amplitude at the center of the spectrum of the signal distribution over a window calculated from the Fourier transform frequency and amplitude information. -Science Direct')
st.markdown("- " + 'Spectral bandwidth is defined as the band width of light at one-half the peak maximum. -perseena')
st.markdown("- " + 'Spectral rolloff is the frequency below which a specified percentage of the total spectral energy. -Music Information Retrieval')

st.write("These features are calculated over a small window of time creating an unknown number of features based on the length of the audio sample. To reduce this we take the mean and standard deviation of each feature. This creates a stable number of features, two for each MFCC and two for each of the other three features.The dataframe is as follows with \'label\' being our target.")
S_df = pd.DataFrame(columns = ["label", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
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
st.dataframe(S_df)

st.write("KDE Plots can help us understand if our features will provide enough significance to make a prediciton. What we are looking for is significant seperation in the peaks.")

with st.expander("KDE Plots"):
    image = Image.open('KDEPlot.png')
    st.image(image, caption='KDE Plots of MFCC\'s')
    image = Image.open('KDEPlotOtherFeatures.png')
    st.image(image, caption='KDE Plots of Other Features\'s')
    
st.write("Unfortunately there is not significant spreading, the model will have a difficult time finding correlation between our features and labels.")
#%% Deciding on a model
st.header("Choosing a Model")
st.write("Model selection was done by using various libraries including, autosklearn, TensorFlow and XGBoost. These are just the top performers many other models were made.")

with st.expander("Autosk-learn Ensemble"):
    image = Image.open('autosklearnensemble.png')
    st.image(image, caption='Confusion Matrix')
    st.write("F1 Scores in order of matrix: 0.68456376, 0.65060241, 0.74914089, 0.7483871, 0.62564103, 0.81735479")
    st.write("Accuracy: 77%")
with st.expander("XGBoost"):
    image = Image.open('XGBoost.png')
    st.image(image, caption='Confusion Matrix')
    st.write("F1 Scores in order of matrix: 0.72839506, 0.74157303, 0.75081967, 0.76300578, 0.71171171, 0.83996995")
    st.write("Accuracy: 79%")
    st.write("The XGBoost model was trained using the best parameters found in a search space from a bayesian search in HyperOpt.")
with st.expander("TensorFlow"):
    image = Image.open('tf.png')
    st.image(image, caption='Confusion Matrix')
    st.write("F1 Scores in order of matrix: 0.869048, 0.730769, 0.802632, 0.771739, 0.786885, 0.860759")
    st.write("Accuracy: 82.5%")
    st.write("The TensorFlow model parameters were found mostly through trial and error. Due to the low complexity of the features, not time based or n-dimensional input, this was possible.")
st.write("The overall best performer was the TensorFlow DNN. It has been chosen as the model of choice and can be run on your voice/files below.")
#%% The model
st.header("Predicting Accent on your Audio")
with st.expander("Using the Model"):
    uploaded_file = st.file_uploader("Choose a file",type = 'mp3')
    
    Models = ModelAssets()
    if uploaded_file is not None:
        Models.createFeatures(uploaded_file)
        st.subheader("Your sample appears to have an accent from: " + Models.predictWithDL()[2:-2].upper())
    
    st.subheader("Record Audio to Predict on")
    st.write("Click and release button to record. It will record as long as there is significant audio coming in.")
    audio_bytes = audio_recorder()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        audio_bytes =io.BytesIO(audio_bytes)
        Models.createFeatures(audio_bytes)
        st.subheader("Your sample appears to have an accent from: " + Models.predictWithDL()[2:-2].upper())