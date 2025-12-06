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
import itertools # <--- Added for list filtering

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Fashion Stylist", page_icon="👗", layout="wide")

# --- SMART PATH SELECTION ---
if os.path.exists("images"):
    folder_path = "images" 
elif os.path.exists("images_sample"):
    folder_path = "images_sample" 
else:
    folder_path = "images_small"

# --- LOAD DATA (WITH NAN FIX) ---
@st.cache_resource
def load_data():
    feature_list = pickle.load(open('features_embedding.pkl', 'rb'))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
    df = pd.read_csv('my_fashion_data.csv')
    
    # 1. Prepare CSV index
    df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
    df = df.set_index('image')
    
    # 2. Reindex to match the pickle filenames
    # This might create NaN (empty) rows if a file isn't in the CSV
    df = df.reindex(filenames)
    
    # 3. CRITICAL FIX: Identify and Drop Broken Rows
    # check which rows have valid data (are not NaN)
    valid_mask = df['subCategory'].notna()
    
    # Filter the DF to keep only valid rows
    df = df[valid_mask].reset_index()
    
    # Filter the Lists to match the valid rows exactly
    # itertools.compress picks items from the list where the mask is True
    feature_list = list(itertools.compress(feature_list, valid_mask))
    filenames = list(itertools.compress(filenames, valid_mask))
    
    return feature_list, filenames, df

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    return model

# --- TRAIN CLASSIFIERS ---
@st.cache_resource
def train_classifiers(feature_list, df):
    # Now that we cleaned the data, these fit() calls will not crash
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
            input_features = extract_features_from_upload(uploaded_file, base_model)
            ai_category = category_classifier.predict([input_features])[0]
            ai_gender = gender_classifier.predict([input_features])[0]
        
        st.write("🤖 **AI Prediction:**")
        
        gender_options = ['Men', 'Women', 'Boys', 'Girls', 'Unisex']
        try:
            default_gender_ix = gender_options.index(ai_gender)
        except:
            default_gender_ix = 0
            
        selected_gender = st.selectbox("Gender", gender_options, index=default_gender_ix)
        
        category_options = ['Topwear', 'Bottomwear', 'Footwear']
        try:
            default_cat_ix = category_options.index(ai_category)
        except:
            default_cat_ix = 0
            
        selected_category = st.selectbox("Category", category_options, index=default_cat_ix)

        compatibility_map = {
            'Topwear': 'Bottomwear',
            'Bottomwear': 'Topwear',
            'Footwear': 'Topwear',
            'Bags': 'Topwear'
        }
        target_category = compatibility_map.get(selected_category, selected_category)
        
        st.info(f"Searching for **{selected_gender}'s {target_category}** matches...")

    st.divider()
    st.subheader("2. Complete the Look")
    
    target_df = df[
        (df['subCategory'] == target_category) & 
        (df['gender'] == selected_gender)
    ].reset_index(drop=True)
    
    if len(target_df) > 0:
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
                    img_path = os.path.join(folder_path, img_name)
                    
                    with col:
                        if os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                            st.caption(f"{selected_gender} {target_category}")
                        else:
                            pass
        else:
            st.warning("No items found.")
    else:
        st.warning(f"No {selected_gender} {target_category} found in database!")

else:
    st.info("👈 Upload an image to start!")
