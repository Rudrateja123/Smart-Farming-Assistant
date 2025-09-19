import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import requests
from PIL import Image

# --- CSS TO HIDE THE FULLSCREEN BUTTON ---
st.markdown(
    """
    <style>
    div[data-testid="stImageToolbar"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- MODEL LOADING ---
@st.cache_resource
def load_disease_model():
    model = tf.keras.models.load_model("models/trained_plant_disease_model.keras")
    return model

@st.cache_resource
def load_crop_recommendation_model():
    with open('models/RandomForest.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_resource
def load_fertilizer_recommendation_model():
    with open('models/Fertilizer_rec.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load all the models
disease_model = load_disease_model()
crop_rec_model = load_crop_recommendation_model()
fertilizer_rec_model = load_fertilizer_recommendation_model()

# --- PREDICTION & UTILITY FUNCTIONS ---
def predict_disease(image_to_predict):
    image = tf.keras.preprocessing.image.load_img(image_to_predict, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = disease_model.predict(input_arr)
    return np.argmax(predictions)

def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    prediction = crop_rec_model.predict(input_data)
    return prediction[0]

def get_remedy(disease_name):
    remedies = {
        'Apple___Apple_scab': "Apply fungicides like captan or mancozeb. Prune infected leaves and branches during dormancy.",'Apple___Black_rot': "Prune out dead or diseased branches. Apply a fungicide spray schedule starting at bud break.",'Apple___Cedar_apple_rust': "Apply fungicides such as myclobutanil or triadimefon. Remove nearby juniper trees if possible.",'Blueberry___healthy': "No remedy needed.",'Cherry_(including_sour)___Powdery_mildew': "Apply sulfur-based or potassium bicarbonate fungicides.",'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Use resistant hybrids. Apply fungicides like pyraclostrobin.",'Corn_(maize)___Common_rust_': "Plant rust-resistant hybrids. Apply fungicides if detected early.",'Corn_(maize)___Northern_Leaf_Blight': "Plant resistant hybrids. Tillage to bury crop residue can help.",'Grape___Black_rot': "Apply fungicides like mancozeb or captan.",'Grape___Esca_(Black_Measles)': "Prune out infected wood. No effective chemical control available.",'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Apply fungicides. Improve air circulation.",'Orange___Haunglongbing_(Citrus_greening)': "No cure. Remove infected trees. Control the Asian citrus psyllid vector.",'Peach___Bacterial_spot': "Apply copper-based bactericides during dormancy.",'Pepper,_bell___Bacterial_spot': "Use disease-free seeds. Apply copper-based sprays.",'Potato___Early_blight': "Apply fungicides like chlorothalonil or mancozeb.",'Potato___Late_blight': "Apply fungicides proactively. Destroy infected plants.",'Squash___Powdery_mildew': "Apply fungicides like sulfur or potassium bicarbonate.",'Strawberry___Leaf_scorch': "Apply fungicides. Renovate strawberry beds after harvest.",'Tomato___Bacterial_spot': "Use disease-free seed and transplants. Apply copper sprays.",'Tomato___Early_blight': "Apply fungicides such as chlorothalonil. Mulch plants.",'Tomato___Late_blight': "Apply fungicides proactively. Avoid overhead watering.",'Tomato___Leaf_Mold': "Ensure good ventilation. Apply fungicides if necessary.",'Tomato___Septoria_leaf_spot': "Apply fungicides like chlorothalonil. Remove lower infected leaves.",'Tomato___Spider_mites Two-spotted_spider_mite': "Apply miticides or insecticidal soaps.",'Tomato___Target_Spot': "Improve air circulation. Apply fungicides.",'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Control whitefly vectors with insecticides.",'Tomato___Tomato_mosaic_virus': "No chemical cure. Remove and destroy infected plants."
    }
    return remedies.get(disease_name, "Maintain good plant health and monitor closely.")

def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
    response = requests.get(complete_url)
    data = response.json()
    if data["cod"] != "404":
        main, wind, weather_desc = data["main"], data["wind"], data["weather"][0]["description"]
        return {"temperature": main["temp"], "humidity": main["humidity"], "wind_speed": wind["speed"], "description": weather_desc.capitalize()}
    else:
        return None

# --- STREAMLIT APP ---
st.sidebar.title("Smart Farming Assistant Navigator")
app_mode = st.sidebar.selectbox("Choose a Feature", ["Home", "Crop Recommendation", "Disease Recognition", "Fertilizer Recommendation", "Weather Forecast"])

if app_mode == "Home":
    st.markdown("<h1 style='text-align: center;'>Smart Farming Assistant</h1>", unsafe_allow_html=True)
    st.image("images/farming.jpg", use_container_width=True)
    st.markdown("---")
    st.markdown("""
    ### Welcome to Your Digital Agronomist! 🌱
    This platform empowers farmers with AI-driven insights for smarter decisions and sustainable agriculture.
    #### **What We Offer:**
    * **🌿 Crop Recommendation:** Get suggestions for the most suitable crop.
    * **🔬 Disease Recognition:** Instantly diagnose plant diseases from a leaf photo.
    * **🧪 Fertilizer Recommendation:** Find the right fertilizer for your crop and soil.
    * **☀️ Weather Forecast:** Get real-time weather updates to plan activities.
    Use the **Navigator** on the left to get started!
    """)

elif app_mode == "Crop Recommendation":
    st.markdown("<h1 style='text-align: center;'>Smart Crop Recommendation</h1>", unsafe_allow_html=True)
    st.image("images/crop-planning.jpg", use_container_width=True)
    st.markdown("### Find the Perfect Crop for Your Land")
    with st.expander("ℹ️ How to Measure These Parameters (For Beginners)"):
        st.markdown("""
        #### *1. Nitrogen (N), Phosphorus (P), Potassium (K) & pH Level*
        These are measured through a *soil test*. You can collect a soil sample from your field and send it to a local agricultural lab. They will send you a report with the exact values for N, P, K (in kg/ha), and the pH level. Alternatively, you can buy a simple digital soil testing kit online.
        
        *Note for testing:* A common range for *Nitrogen* is 20-120 kg/ha. For *Phosphorus, 10-80 kg/ha. For **Potassium, 10-90 kg/ha. A neutral **pH* is around 6.5-7.5.
        
        #### *2. Temperature & Humidity*
        These can be measured using a simple *digital weather station* or even a home weather device placed in a shaded area in your field. For an average value, you can also check your local weather forecast service (like the India Meteorological Department, IMD).
        
        *Note for testing:* A common growing *Temperature* is between 18-35°C. A typical *Humidity* is between 60-90%.
        
        #### *3. Rainfall*
        This is measured with a *rain gauge*, which is a simple calibrated cylinder that collects rain. You can buy one for your farm to measure the exact rainfall in millimeters (mm). For an average value, you can refer to historical weather data for your region.
        
        *Note for testing:* Annual *Rainfall* for many crops ranges from 40mm to 200mm. Try a value like 100mm.
        """)
    st.sidebar.header("Enter Soil & Climate Details")
    nitrogen, phosphorus, potassium = st.sidebar.number_input("Nitrogen (N) in kg/ha", 0.0, 140.0, 0.0), st.sidebar.number_input("Phosphorus (P) in kg/ha", 0.0, 145.0, 0.0), st.sidebar.number_input("Potassium (K) in kg/ha", 0.0, 205.0, 0.0)
    temperature, humidity, ph, rainfall = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 0.0), st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 0.0), st.sidebar.number_input("pH Level", 0.0, 14.0, 0.0), st.sidebar.number_input("Rainfall (mm)", 0.0, 300.0, 0.0)
    if st.sidebar.button("Recommend Crop"):
        if all(v == 0.0 for v in [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]):
            st.warning("Please enter values before predicting.")
        else:
            prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            st.success(f"**✅ The recommended crop is: {prediction.capitalize()}**")

elif app_mode == "Disease Recognition":
    st.markdown("<h1 style='text-align: center;'>Smart Disease Recognition</h1>", unsafe_allow_html=True)
    st.image("images/disease-detect.jpg", use_container_width=True)
    st.markdown("### Instantly Diagnose Plant Diseases")
    test_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    if test_image:
        st.image(test_image, caption="Uploaded Image", width=400)
        if st.button("Predict Disease"):
            st.snow()
            with st.spinner('Processing image...'):
                result_index = predict_disease(test_image)
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
                predicted_disease_name = class_name[result_index]
            st.success(f"**Diagnosis: {predicted_disease_name.replace('___', ' - ')}**")
            if 'healthy' not in predicted_disease_name:
                remedy = get_remedy(predicted_disease_name)
                st.markdown("---")
                st.markdown("### 💊 Recommended Remedy:")
                st.info(remedy)

elif app_mode == "Fertilizer Recommendation":
    st.markdown("<h1 style='text-align: center;'>Smart Fertilizer Recommendation</h1>", unsafe_allow_html=True)
    st.image("images/fertilizer.jpg", use_container_width=True)
    st.markdown("### Find the Right Fertilizer for Your Crop")
    
    st.sidebar.header("Enter Details for Fertilizer Recommendation")
    
    # --- FIXED HERE: All default values are now 0.0 ---
    temp, humi, mois = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 0.0), st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 0.0), st.sidebar.number_input("Moisture", 0.0, 100.0, 0.0)
    soil = st.sidebar.selectbox("Soil Type", ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'])
    crop = st.sidebar.selectbox("Crop Type", ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts'])
    nitro, pota, phos = st.sidebar.number_input("Nitrogen (N)", 0.0, 100.0, 0.0), st.sidebar.number_input("Potassium (K)", 0.0, 100.0, 0.0), st.sidebar.number_input("Phosphorous (P)", 0.0, 100.0, 0.0)

    if st.sidebar.button("Recommend Fertilizer"):
        if temp == 0.0 and humi == 0.0 and mois == 0.0 and nitro == 0.0 and pota == 0.0 and phos == 0.0:
            st.warning("Please enter values before predicting.")
        else:
            fertilizer_names_list = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP', 'Urea']
            
            all_columns = ['Temparature', 'Humidity ', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil Type_Clayey', 'Soil Type_Loamy', 'Soil Type_Red', 'Soil Type_Sandy', 'Crop Type_Cotton', 'Crop Type_Ground Nuts', 'Crop Type_Maize', 'Crop Type_Millets', 'Crop Type_Oil seeds', 'Crop Type_Paddy', 'Crop Type_Pulses', 'Crop Type_Sugarcane', 'Crop Type_Tobacco', 'Crop Type_Wheat']
            
            input_df = pd.DataFrame(columns=all_columns)
            input_df.loc[0] = 0
            
            input_df['Temparature'] = temp
            input_df['Humidity '] = humi
            input_df['Moisture'] = mois
            input_df['Nitrogen'] = nitro
            input_df['Potassium'] = pota
            input_df['Phosphorous'] = phos
            
            soil_col_name, crop_col_name = f'Soil Type_{soil}', f'Crop Type_{crop}'
            if soil_col_name in all_columns: input_df[soil_col_name] = 1
            if crop_col_name in all_columns: input_df[crop_col_name] = 1
            
            final_input = input_df[all_columns]
            prediction_index = fertilizer_rec_model.predict(final_input)
            predicted_fertilizer_name = fertilizer_names_list[prediction_index[0]]
            
            st.success(f"**✅ The recommended fertilizer is: {predicted_fertilizer_name}**")

elif app_mode == "Weather Forecast":
    st.markdown("<h1 style='text-align: center;'>Weather Forecast</h1>", unsafe_allow_html=True)
    st.image("images/weather.jpg", use_container_width=True)
    st.markdown("### Get Real-Time Weather Updates for Your Location")
    api_key = st.secrets["OPENWEATHERMAP_API_KEY"]    
    city = st.text_input("Enter your city name (e.g., Visakhapatnam):")
    if st.button("Get Forecast"):
        if city:
            with st.spinner(f"Fetching weather for {city}..."):
                weather_data = get_weather(city, api_key)
                if weather_data:
                    st.success(f"**Weather in {city.capitalize()}:**")
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Temperature", f"{weather_data['temperature']} °C")
                    with col2: st.metric("Humidity", f"{weather_data['humidity']}%")
                    with col3: st.metric("Wind Speed", f"{weather_data['wind_speed']} m/s")
                    st.info(f"**Condition:** {weather_data['description']}")
                else:
                    st.error("City not found. Please check the spelling.")
        else:
            st.warning("Please enter a city name.")