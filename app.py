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

# --- TRAIN CLASSIFIERS (THE NEW MINI BRAINS) ---
@st.cache_resource
def train_classifiers(feature_list, df):
    # We train two small models instantly when the app loads
    # Model 1: Guesses if it's Top/Bottom/Footwear
    category_model = KNeighborsClassifier(n_neighbors=5)
    category_model.fit(feature_list, df['subCategory'])
    
    # Model 2: Guesses if it's Men/Women
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

# --- FUNCTION: RECOMMENDATION LOGIC ---
def recommend(features, df, feature_list, category_classifier, gender_classifier):
    # 1. PREDICT details about the uploaded image (The "Magic" Step)
    # We reshape features to look like a list of 1 item
    predicted_category = category_classifier.predict([features])[0]
    predicted_gender = gender_classifier.predict([features])[0]
    
    st.write(f"🤖 **AI Analysis:** I think this is **{predicted_gender}'s {predicted_category}**.")
    
    # 2. Define Compatibility Rules (Shirt -> Pants)
    compatibility_map = {
        'Topwear': 'Bottomwear',
        'Bottomwear': 'Topwear',
        'Footwear': 'Topwear',
        'Bags': 'Topwear'
    }
    
    # If the rule exists, switch category. If not (e.g., Watches), stick to visual search
    target_category = compatibility_map.get(predicted_category, predicted_category)
    
    # 3. Filter the Dataset
    target_df = df[
        (df['subCategory'] == target_category) & 
        (df['gender'] == predicted_gender)
    ].reset_index(drop=True)
    
    if len(target_df) == 0:
        st.warning(f"No matching items found for {predicted_gender} {target_category}.")
        return []

    # 4. Get features for filtered items
    target_features = []
    target_filenames = []
    
    # Map back to original features (Simple loop for MVP)
    # Note: In a pro app, you'd pre-calculate these indices to make it faster
    for filename in target_df['image']:
        try:
            original_index = df[df['image'] == filename].index[0]
            target_features.append(feature_list[original_index])
            target_filenames.append(filename)
        except:
            pass
            
    if len(target_features) == 0:
        return []

    # 5. Find Neighbors
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(target_features)
    distances, indices = neighbors.kneighbors([features])
    
    return [target_filenames[i] for i in indices[0]]

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
        
        # --- THE FIX: MANUAL OVERRIDE SECTION ---
        st.write("🤖 **AI Prediction:**")
        
        # We use the AI's prediction as the 'index' (default value) for the dropdowns
        # This gives the user control to fix mistakes
        
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
        
       

    # --- RECOMMENDATION LOGIC (Updated to use SELECTED values) ---
    st.divider()
    st.subheader("2. Complete the Look")
    
    # Define Rules (Shirt -> Pants)
    compatibility_map = {
        'Topwear': 'Bottomwear',
        'Bottomwear': 'Topwear',
        'Footwear': 'Topwear',
        'Bags': 'Topwear'
    }
    target_category = compatibility_map.get(selected_category, selected_category)
    
    # Filter Dataset based on USER SELECTION (Not just AI guess)
    target_df = df[
        (df['subCategory'] == target_category) & 
        (df['gender'] == selected_gender)
    ].reset_index(drop=True)
    
    if len(target_df) > 0:
        # ... (Insert your standard Nearest Neighbors logic here) ...
        # Get features for filtered items
        target_features = []
        target_filenames = []
        
        for filename in target_df['image']:
            try:
                original_index = df[df['image'] == filename].index[0]
                target_features.append(feature_list[original_index])
                target_filenames.append(filename)
            except:
                pass
        
        if len(target_features) > 0:
            neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
            neighbors.fit(target_features)
            distances, indices = neighbors.kneighbors([input_features])
            
            cols = st.columns(5)
            for i, col in enumerate(cols):
                if i < len(indices[0]):
                    idx = indices[0][i]
                    img_name = target_filenames[idx]
                    img_path = os.path.join("images_small", img_name)
                    with col:
                        st.image(img_path, use_container_width=True)
                        st.caption(f"{selected_gender} {target_category}")
        else:
            st.warning("No items found.")
    else:
        st.warning(f"No {selected_gender} {target_category} found in database!")

else:
    st.info("👈 Upload an image to start!")