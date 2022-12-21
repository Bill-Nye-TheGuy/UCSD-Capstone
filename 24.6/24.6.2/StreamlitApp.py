# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:34:33 2022

@author: darth
"""

import streamlit as st
from EndToEndRefactor import ModelAssets

st.title("Audio Accent Identifier")
uploaded_file = st.file_uploader("Choose a file",type = 'mp3')
Models = ModelAssets()
if uploaded_file is not None:
    z = Models.predictOnFile(uploaded_file)
    st.dataframe(z)
