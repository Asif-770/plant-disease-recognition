import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd



#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element     

#Sidebar
st.sidebar.title("DASHBOARD")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    # image_path = "plantimage.jpeg"
    # st.image(image_path,use_column_width=True)
    
#     st.markdown(
#     """
#     <style>
#         back
#     st.write()
#     </style>
#     """,
#     unsafe_allow_html=True
# )


    st.markdown("""
    
    
       
    
    
    Welcome to the Plant Disease Recognition System! 
    ### USING CNN MACHINE LEARNING MODEL: 
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    # enable = st.checkbox ("Enable camera")
    # test_image= st.camera_input("Take a picture",disabled=not enable)

    # if test_image is not None:
    #     # To read image file buffer as a 3D uint8 tensor with TensorFlow:
    #     bytes_data = test_image.getvalue()
    #     img_tensor = tf.io.decode_image(bytes_data, channels=3)
    
    

    # if test_image:
    #     st.image(test_image)
    
    #Predict button
    if(st.button("Predict")):
        
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
        st.balloons()
        
            
    st.write('To Find Possible Cure Of This Disease Please Click Below button')
    if(st.button('Cure')):  
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
    

               
    st.select_slider("Rate This Model",options=['Good','Better','Best'])
    # audio_value = st.experimental_audio_input("Tell us about our Machine Learning Model")



    # if audio_value:
    #     st.audio(audio_value)