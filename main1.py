import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# Model Prediction Function
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
st.sidebar.title("🌿 Plant Disease Dashboard")
app_mode = st.sidebar.radio("Go to", ["Home", "Model Summary", "Disease Recognition", "About"])

# Banner



# ================= HOME =================
if app_mode == "Home":

    import base64

    # Encode image to base64
    def get_base64(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    img_base64 = get_base64("bn1.jpeg")

    # Inject CSS background
    st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(
            rgba(0, 0, 0, 0.6), 
            rgba(0, 0, 0, 0.6)
        ), url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


    # Styled Homepage Text Content
    st.markdown(
        """
        <style>
            .title {
                font-size: 30px;
                color: #4CAF50;
                font-weight: bold;
                text-align: center;
            }
            .subtitle {
                font-size: 22px;
                color: #4CAF50;
                margin-top: 20px;
                font-weight: 600;
            }
            .text, ul.text li {
                font-size: 16px;
                color: white;
                line-height: 1.6;
            }
            body {
                background-color: #1e1e1e;
            }
        </style>

        <div class="title">🌿 PLANT DISEASE RECOGNITION SYSTEM</div>

        <p class="text"><strong>Welcome to the Plant Disease Recognition System!</strong></p>

        <div class="subtitle">🚀 USING CNN MACHINE LEARNING MODEL:</div>
        <p class="text">
            Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
        </p>

        <div class="subtitle">🔎 How It Works</div>
        <ul class="text">
            <li><strong>Upload Image:</strong> Go to the <em>Disease Recognition</em> page and upload an image of a plant with suspected diseases.</li>
            <li><strong>Analysis:</strong> Our system processes the image using advanced CNN algorithms.</li>
            <li><strong>Results:</strong> View the predicted disease and treatment advice.</li>
        </ul>

        <div class="subtitle">🎯 Why Choose Us?</div>
        <ul class="text">
            <li><strong>Accuracy:</strong> State-of-the-art ML ensures accurate disease detection.</li>
            <li><strong>User-Friendly:</strong> Intuitive interface for all users.</li>
            <li><strong>Fast:</strong> Get results in seconds.</li>
        </ul>

        <div class="subtitle">🌱 Get Started</div>
        <p class="text">
            Click on the <strong>Disease Recognition</strong> page in the sidebar to upload an image and experience the power of our system.
        </p>
        """,
        unsafe_allow_html=True
    )

# ================= UPLOAD IMAGE =================
# elif app_mode == "Upload Image":
#     st.subheader("📤 Upload an Image")
#     uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])
#     if uploaded_file:
#         st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
#         st.success("Image uploaded successfully!")

# ================= MODEL SUMMARY =================
# ================= MODEL SUMMARY =================
elif app_mode == "Model Summary":
    import base64

    # Encode image to base64
    def get_base64(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    img_base64 = get_base64("bn1.jpeg")

    # Inject CSS background
    st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(
            rgba(0, 0, 0, 0.6), 
            rgba(0, 0, 0, 0.6)
        ), url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

    
    st.subheader("📋 CNN Model Architecture & Accuracy")

    st.markdown("### 🧠 Model: `sequential_2`")
    
    st.code("""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 128, 128, 32)   │           896 │
│ conv2d_1 (Conv2D)               │ (None, 126, 126, 32)   │         9,248 │
│ max_pooling2d (MaxPooling2D)    │ (None, 63, 63, 32)     │             0 │
│ conv2d_2 (Conv2D)               │ (None, 63, 63, 64)     │        18,496 │
│ conv2d_3 (Conv2D)               │ (None, 61, 61, 64)     │        36,928 │
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 30, 30, 64)     │             0 │
│ conv2d_4 (Conv2D)               │ (None, 30, 30, 128)    │        73,856 │
│ conv2d_5 (Conv2D)               │ (None, 28, 28, 128)    │       147,584 │
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 14, 14, 128)    │             0 │
│ conv2d_6 (Conv2D)               │ (None, 14, 14, 264)    │       304,392 │
│ conv2d_7 (Conv2D)               │ (None, 12, 12, 264)    │       627,528 │
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 6, 6, 264)      │             0 │
│ conv2d_8 (Conv2D)               │ (None, 6, 6, 512)      │     1,217,024 │
│ conv2d_9 (Conv2D)               │ (None, 4, 4, 512)      │     2,359,808 │
│ max_pooling2d_4 (MaxPooling2D)  │ (None, 2, 2, 512)      │             0 │
│ dropout (Dropout)               │ (None, 2, 2, 512)      │             0 │
│ flatten (Flatten)               │ (None, 2048)           │             0 │
│ dense (Dense)                   │ (None, 1500)           │     3,073,500 │
│ dropout_1 (Dropout)             │ (None, 1500)           │             0 │
│ dense_1 (Dense)                 │ (None, 38)             │        57,038 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

🧮 Total Parameters: 7,926,298 (~30.24 MB)
🛠️ Trainable Parameters: 7,926,298
❌ Non-trainable Parameters: 0
    """, language="text")

    st.markdown("### 📊 Training & Validation Results")
    st.write("✅ **Training Accuracy**: `97.64%`")
    st.write("✅ **Validation Accuracy**: `94.06%`")
    st.write("🗓️ Trained for **10 Epochs** using Adam optimizer and Categorical Crossentropy loss.")


# ================= DISEASE RECOGNITION =================
elif app_mode == "Disease Recognition":
    import base64

    # Encode image to base64
    def get_base64(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    img_base64 = get_base64("bn1.jpeg")

    # Inject CSS background
    st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(
            rgba(0, 0, 0, 0.6), 
            rgba(0, 0, 0, 0.6)
        ), url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

    st.image("bn.png", use_column_width=True)
    st.subheader("🧠 Plant Disease Recognition")
    test_image = st.file_uploader("Choose an image of a plant leaf")

    if st.button("Show Image") and test_image:
        st.image(test_image, use_column_width=True)

    if st.button("Predict") and test_image:
        result_index = model_prediction(test_image)

        class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                       'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                       'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                       'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                       'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                       'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                       'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                       'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                       'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                       'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                       'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                       'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                       'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                       'Tomato___healthy']

        prediction = class_names[result_index]
        st.success(f"🌿 The model predicts: **{prediction}**")
        st.balloons()

        # === TREATMENT LOGIC (inserted here for maintainability) ===
    st.info("🔍 To see treatment/cure information, click the button below:")
    if st.button("Show Cure"):
        result_index = model_prediction(test_image)
        if (result_index==0):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Venturia inaequalis.",
                                    "Use fungicides like captan, mancozeb, or myclobutanil. Apply sprays during spring, starting at bud break.",
                                    "Remove fallen leaves and debris, prune trees for better airflow, and use resistant apple varieties if available."]
                    }

                )
            )
        elif(result_index==1):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Botryosphaeria obtusa.",
                                    "Use fungicides such as copper or captan. Apply before rain and after severe storms.",
                                    "Prune out infected branches, remove mummified fruit, and avoid injuring the bark."]
                    }

                )
            )
        elif(result_index==2):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Gymnosporangium juniperi-virginianae.",
                                        "Apply fungicides (e.g., myclobutanil) early in the season.",
                                        "Remove nearby cedar or juniper plants, as they can harbor the fungus."]

                    }

                )
            )
        elif(result_index==3):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Prevention"],
                        "Statements":["Regular inspections, proper pruning, and timely fungicide application can keep trees healthy."]

                    }

                )
            )
        elif(result_index==4):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Prevention"],
                        "Statements":["Maintain proper soil acidity (pH 4.5–5.5), water adequately, and use mulch to keep weeds down."]

                    }

                )
            )
        elif(result_index==5):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungal pathogens such as Podosphaera clandestina.",
                                        "Use fungicides like sulfur or neem oil; start when buds open.",
                                    "Improve air circulation by pruning, and avoid wetting foliage during irrigation."]

                    }

                )
            )
        elif(result_index==6):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Prevention"],
                        "Statements":["Proper spacing, balanced fertilization, and regular monitoring keep cherry trees healthy."]

                    }

                )
            )
        elif(result_index==7):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Cercospora zeae-maydis.",
                                        "Use fungicides containing strobilurins or triazoles.",
                                            "Rotate crops, remove corn debris after harvest, and select resistant varieties."]

                    }

                )
            )
        elif(result_index==8):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Puccinia sorghi.",
                                    " Apply fungicides such as azoxystrobin.",
                                        "Plant rust-resistant hybrids and avoid high planting densities."]

                    }

                )
            )
        elif(result_index==9):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Exserohilum turcicum.",
                                        "Use fungicides like mancozeb or chlorothalonil.",
                                        "Crop rotation and resistant hybrids help reduce the risk."]

                    }

                )
            )
        elif(result_index==10):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Prevention"],
                        "Statements":["Ensure optimal fertilization, soil drainage, and weed control."]

                    }

                )
            )
        elif(result_index==11):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":[" Fungus Guignardia bidwellii.",
                                        "Use fungicides (mancozeb or myclobutanil).",
                                        "Remove infected fruit and leaves, prune vines for airflow, and use resistant cultivars."]

                    }

                )
            )
        elif(result_index==12):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":[" Fungi in the Phaeoacremonium and Phaeomoniella species.",
                                        "Avoid over-watering and pruning in wet conditions.",
                                    "Good vineyard sanitation and balanced fertilization."]

                    }

                )
            )
        elif(result_index==13):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Isariopsis clavispora.",
                                    "Fungicides like copper-based sprays.",
                                        "Remove infected leaves, and avoid overhead irrigation."]

                    }

                )
            )
        elif(result_index==14):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Prevention"],
                        "Statements":["Regular pruning, good airflow, and careful watering practices."]

                    }

                )
            )
        elif(result_index==15):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Bacteria Candidatus Liberibacter spp., spread by Asian citrus psyllid.",
                                        "No cure; control psyllids with insecticides like imidacloprid.",
                                        "Regularly monitor for psyllids and use resistant rootstocks."]

                    }

                )
            )
        elif(result_index==16):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Bacterium Xanthomonas arboricola pv. pruni.",
                                        "Use copper-based sprays early in the season.",
                                        "Plant resistant varieties and avoid overhead watering."]

                    }

                )
            )
        elif(result_index==17):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Prevention"],
                        "Statements":["Proper pruning, disease-resistant varieties, and preventive sprays."]

                    }

                )
            )
        elif(result_index==18):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Xanthomonas campestris.",
                                        "Copper-based bactericides.",
                                        "Use certified disease-free seeds and avoid overhead watering."]

                    }

                )
            )
        elif(result_index==19):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Prevention"],
                        "Statements":["Rotate crops, maintain good soil health, and water at the base."]

                    }

                )
            )
        elif(result_index==20):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":[" Fungus Alternaria solani.",
                                        "Use fungicides such as chlorothalonil.",
                                        "Rotate crops, avoid planting potatoes next to tomatoes, and keep foliage dry."]

                    }

                )
            )
        elif(result_index==21):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Phytophthora infestans.",
                                        "Use fungicides like mancozeb or copper.",
                                        "Rotate crops, plant disease-free tubers, and space plants for airflow."]

                    }

                )
            )
        elif(result_index==22):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Prevention"],
                        "Statements":["Monitor for disease regularly and keep weeds under control."]

                    }

                )
            )
        elif(result_index==23):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Prevention"],
                        "Statements":["Proper soil drainage, removing old canes, and controlling pests."]

                    }

                )
            )
        elif(result_index==24):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Prevention"],
                        "Statements":["Rotate crops, maintain soil health, and monitor for pests and diseases."]

                    }

                )
            )
        elif(result_index==25):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Podosphaera xanthii.",
                                        "Use fungicides like sulfur or potassium bicarbonate.",
                                        "Avoid dense plantings and remove infected leaves."]

                    }

                )
            )
        elif(result_index==26):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Diplocarpon earliana.",
                                        "Use fungicides such as myclobutanil.",
                                        "Avoid wetting the leaves and improve air circulation."]

                    }

                )
            )
        elif(result_index==27):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Prevention"],
                        "Statements":["Regular inspections, good drainage, and maintaining proper spacing."]

                    }

                )
            )
        elif(result_index==28):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Xanthomonas campestris.",
                                    "Copper-based bactericides.",
                                    "Use certified seeds, avoid working in wet fields, and rotate crops."]

                    }

                )
            )
        elif(result_index==29):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Alternaria solani.",
                                        "Fungicides like chlorothalonil.",
                                        "Crop rotation and proper spacing."]

                    }

                )
            )
        elif(result_index==30):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Phytophthora infestans.",
                                       "Fungicides containing mancozeb or copper.",
                                        "Use resistant varieties, rotate crops, and control weeds."]

                    }

                )
            )
        elif(result_index==31):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Cladosporium fulvum.",
                                       "Fungicides like copper-based ones.",
                                        "Improve air circulation and avoid high humidity."]

                    }

                )
            )
        elif(result_index==32):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Septoria lycopersici.",
                                       "Use fungicides like chlorothalonil.",
                                        "Use drip irrigation and rotate crops."]

                    }

                )
            )
        elif(result_index==33):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":[" Pest Tetranychus urticae.",
                                        "Use insecticidal soap or miticides.",
                                        "Maintain moderate humidity and inspect regularly."]

                    }

                )
            )
        elif(result_index==34):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Fungus Corynespora cassiicola.",
                                        "Use fungicides containing copper.",
                                        "Rotate crops and prune for airflow."]

                    }

                )
            )
            
            
        elif(result_index==35):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Virus spread by whiteflies.",
                                       "No cure; control whiteflies with insecticides.",
                                        "Use reflective mulches, resistant varieties, and manage whitefly populations."]

                    }

                )
            )
        elif(result_index==36):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Cause","Treatment","Prevention"],
                        "Statements":["Tobacco mosaic virus (TMV), highly contagious, spread by contaminated tools, hands, or seeds.",
                                        "No direct cure for TMV; instead, focus on preventive measures.",
                                        "Soak seeds in a 10% bleach solution or trisodium phosphate (TSP) solution before planting."]

                    }

                )
            )
        elif(result_index==37):
            st.write(
                pd.DataFrame(
                    {
                        "Topics":["Prevention"],
                        "Statements":["Proper Spacing: Space plants adequately to improve air circulation and reduce humidity"]

                    }

                )
            )
        
        
           
            
            

    

    st.select_slider("💬 Rate This Model", options=["Good", "Better", "Best"])

# ================= ABOUT =================
elif app_mode == "About":
    import base64

    # Encode image to base64
    def get_base64(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    img_base64 = get_base64("bn1.jpeg")

    # Inject CSS background
    st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(
            rgba(0, 0, 0, 0.6), 
            rgba(0, 0, 0, 0.6)
        ), url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

    st.subheader("ℹ️ About This Project")
    st.markdown("""
    ### 🧠 Project Overview
    This CNN-based plant disease recognition project is built using:

    - **Python**
    - **TensorFlow**
    - **Streamlit**

    It classifies images into **38 different plant disease classes**  
    and suggests possible **treatments and cures** based on prediction results.

    ### 📂 About Dataset
    This dataset is a recreated version using offline augmentation based on the original dataset.  
    It contains approximately **87,000 RGB images** of both **healthy** and **diseased** crop leaves, spread across **38 categories**.

    - `train/` — 70,295 images  
    - `validation/` — 17,572 images  
    - `test/` — 33 images (for prediction)  
    - Dataset was split using an **80/20 ratio** for training and validation, preserving folder structure.

    ### 🛠️ Skills & Technologies Used
    - Convolutional Neural Networks (CNN)
    - Image Augmentation
    - Deep Learning & Transfer Learning
    - Model Evaluation & Accuracy Tracking
    - Streamlit Web App Development
    - Python Data Handling (NumPy, Pandas)

    ---
    ### 👨‍💻 Developed by: Md Asif Khan  
    📫 Email: ak2130251@gmail.com  
    🌐 GitHub: [Your GitHub Profile](https://github.com/Asif-770)
    """)


# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>© 2025 Md Asif Khan | Plant Disease Recognition | Powered by Streamlit</p>", unsafe_allow_html=True)
