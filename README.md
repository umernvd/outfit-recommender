# 👗 AI Fashion Stylist: Visual Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-MVP%20Complete-success)]()

> **An end-to-end Machine Learning web app that acts as a personal stylist.** > Upload any clothing item, and the AI suggests complete outfit recommendations (e.g., input a *Shirt* $\rightarrow$ get matching *Pants*) using deep learning and compatibility logic.

---


## 🚀 Project Overview

Most e-commerce recommendations rely on text keywords ("Blue shirt"). This project implements **Visual Search** using Computer Vision to understand the *style, texture, and pattern* of a garment.

Beyond simple similarity search, this system features a **Compatibility Logic Layer** that understands fashion rules (Topwear matches with Bottomwear) and includes a **Human-in-the-Loop** workflow to handle edge cases in gender/category classification.

### Key Features
* **🧠 Deep Feature Extraction:** Uses **ResNet50** (pre-trained on ImageNet) to convert images into 2048-dimensional embedding vectors.
* **🔍 Content-Based Retrieval:** Implements **K-Nearest Neighbors (KNN)** to find visually compatible items based on Euclidean distance.
* **👔 Smart Compatibility Rules:** Logic layer that maps input categories to complementary output categories (e.g., *Input: Men's Polo* $\rightarrow$ *Output: Men's Trousers*).
* **🛡️ Human-in-the-Loop UX:** Interactive UI allows users to correct AI misclassifications (e.g., "This is actually a Women's top") in real-time before generating recommendations.
* **⚡ Real-Time Latency:** Optimized for sub-second query performance using Scikit-learn and Streamlit caching.

---


1.  **Input:** User uploads an image via Streamlit.
2.  **Preprocessing:** Image is resized to (224, 224) and normalized.
3.  **Feature Extraction:** Image is passed through **ResNet50** (with top layers removed) to generate a style vector.
4.  **Classification:** Lightweight **KNN Classifiers** predict Gender and Category.
5.  **Filtering:** The dataset is filtered based on business rules (Compatibility Map).
6.  **Retrieval:** The system calculates distances between the input vector and the filtered dataset to return top-N recommendations.

---

## 🛠️ Tech Stack

* **Core Logic:** Python
* **Deep Learning:** TensorFlow (Keras), ResNet50
* **Machine Learning:** Scikit-Learn (KNN, NearestNeighbors)
* **Data Processing:** Pandas, NumPy, Pillow
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Cloud / Local

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/fashion-recommender-system.git](https://github.com/YOUR_USERNAME/fashion-recommender-system.git)
cd fashion-recommender-system

### 2. Install Dependencies
```bash
pip install -r requirements.txt

3. Run the App
Bash

python -m streamlit run app.py
The app will open in your browser at http://localhost:8501.

📂 Project Structure
Plaintext

fashion-recommender/
│
├── app.py                   # Main Streamlit application logic
├── features_embedding.pkl   # Pre-computed image features (Mini-Brain)
├── filenames.pkl            # Corresponding filenames for the features
├── my_fashion_data.csv      # Metadata (Gender, Category, etc.)
├── images_small/            # Sample images for the demo
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
🚧 Future Improvements (Roadmap)
While this MVP works well for demonstration, scaling to production would require:

Vector Database: Migrating from in-memory KNN to FAISS or Pinecone for handling millions of items efficiently.

Object Detection: Implementing YOLO to auto-crop items from user selfies (removing background noise).

Advanced Filtering: Adding filters for Season, Occasion (Formal/Casual), and Color Theory matching.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

📝 License
This project is licensed under the MIT License.

Dataset Credit: Fashion Product Images Dataset on Kaggle.
