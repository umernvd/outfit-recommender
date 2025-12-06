import streamlit as st
import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Fashion Stylist", page_icon="👗", layout="wide")

# --- SMART PATH SELECTION (Fix for Cloud vs Local) ---
# We determine where the images are before we start
if os.path.exists("images"):
    folder_path = "images"  # Local mode (High Res)
elif os.path.exists("images_sample"):
    folder_path = "images_sample" # GitHub mode
else:
    folder_path = "images_small" # Fallback if you named it 'images_small'

# --- LOAD DATA (CACHED) ---
@st.cache_resource
def load_data():
    feature_list = pickle.load(open('features_embedding.pkl', 'rb'))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
    df = pd.read_csv('my_fashion_data.csv')
    return feature_list, filenames, df

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def load_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    return model

# --- TRAIN CLASSIFIERS ---
@st.cache_resource
def train_classifiers(feature_list, df):
    category_model = KNeighborsClassifier(n_neighbors=5)
    category_model.fit(feature_list, df['subCategory'])
    
    gender_model = KNeighborsClassifier(n_neighbors=5)
    gender_model.fit(feature_list, df['gender'])
    
    return category_model, gender_model

# Load everything
feature_list, filenames, df = load_data()
base_model = load_model()
category_classifier, gender_classifier = train_classifiers(feature_list, df)

# --- FUNCTION: EXTRACT FEATURES ---
def extract_features_from_upload(uploaded_file, model):
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    return model.predict(preprocessed_img, verbose=0).flatten()

# --- UI LAYOUT ---
st.title("👗 AI Fashion Stylist")

with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png"])

if uploaded_file is not None:
    st.subheader("1. Analyze Item")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(uploaded_file, use_container_width=True)

    with col2:
        with st.spinner('AI is analyzing...'):
            # 1. Extract Features
            input_features = extract_features_from_upload(uploaded_file, base_model)
            
            # 2. Get AI Predictions
            ai_category = category_classifier.predict([input_features])[0]
            ai_gender = gender_classifier.predict([input_features])[0]
        
        # --- MANUAL OVERRIDE SECTION ---
        st.write("🤖 **AI Prediction:**")
        
        gender_options = ['Men', 'Women', 'Boys', 'Girls', 'Unisex']
        try:
            default_gender_ix = gender_options.index(ai_gender)
        except:
            default_gender_ix = 0
            
        selected_gender = st.selectbox(
            "Gender", 
            gender_options, 
            index=default_gender_ix
        )
        
        category_options = ['Topwear', 'Bottomwear', 'Footwear']
        try:
            default_cat_ix = category_options.index(ai_category)
        except:
            default_cat_ix = 0
            
        selected_category = st.selectbox(
            "Category", 
            category_options, 
            index=default_cat_ix
        )

        # Move Logic Up so we can print the target category
        compatibility_map = {
            'Topwear': 'Bottomwear',
            'Bottomwear': 'Topwear',
            'Footwear': 'Topwear',
            'Bags': 'Topwear'
        }
        target_category = compatibility_map.get(selected_category, selected_category)
        
        st.info(f"Searching for **{selected_gender}'s {target_category}** matches...")

    # --- RECOMMENDATION LOGIC ---
    st.divider()
    st.subheader("2. Complete the Look")
    
    # Filter Dataset based on USER SELECTION
    target_df = df[
        (df['subCategory'] == target_category) & 
        (df['gender'] == selected_gender)
    ].reset_index(drop=True)
    
    if len(target_df) > 0:
        # Get features for filtered items
        target_features = []
        target_filenames = []
        
        for filename in target_df['image']:
            try:
                original_index = df[df['image'] == filename].index[0]
                target_features.append(feature_list[original_index])
                target_filenames.append(