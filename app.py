import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ✅ Base path for artifacts
# base_path = os.path.join(os.path.dirname(__file__), "artifacts")

# ✅ Load model, label encoder, and selected features
# model = joblib.load(os.path.join(base_path, "best_forest_cover_model.pkl"))
# selected_features = joblib.load(os.path.join(base_path, "selected_features.pkl"))
# label_encoder = joblib.load(os.path.join(base_path, "label_encoder.pkl"))

model = joblib.load("best_forest_cover_model.pkl")
selected_features = joblib.load("selected_features.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Title
st.title("🌲 EcoType: Forest Cover Type Prediction")

# Sidebar inputs
st.sidebar.header("Input Cartographic Features")

# Collect user inputs
def user_input_features():
    elevation = st.sidebar.number_input("Elevation (m)", min_value=0, max_value=5000, value=2500)
    slope = st.sidebar.number_input("Slope (degrees)", min_value=0, max_value=90, value=10)
    aspect = st.sidebar.number_input("Aspect (degrees)", min_value=0, max_value=360, value=180)
    hor_dist_hydro = st.sidebar.number_input("Horizontal Distance to Hydrology (m)", value=100)
    vert_dist_hydro = st.sidebar.number_input("Vertical Distance to Hydrology (m)", value=50)
    hor_dist_road = st.sidebar.number_input("Horizontal Distance to Roadways (m)", value=200)
    hillshade_9am = st.sidebar.number_input("Hillshade at 9am", value=220)
    hillshade_noon = st.sidebar.number_input("Hillshade at Noon", value=230)
    hillshade_3pm = st.sidebar.number_input("Hillshade at 3pm", value=210)
    hor_dist_fire = st.sidebar.number_input("Horizontal Distance to Fire Points (m)", value=150)

    # Wilderness Area (one-hot encoded)
    wilderness = st.sidebar.selectbox("Wilderness Area", ["Area 1", "Area 2", "Area 3", "Area 4"])
    wilderness_dict = {"Area 1": [1, 0, 0, 0], "Area 2": [0, 1, 0, 0],
                       "Area 3": [0, 0, 1, 0], "Area 4": [0, 0, 0, 1]}

    # Soil Type (one-hot encoded)
    soil_type = st.sidebar.selectbox("Soil Type", [f"Soil_Type_{i}" for i in range(1, 41)])
    soil_vector = [1 if soil_type == f"Soil_Type_{i}" else 0 for i in range(1, 41)]

    # Derived features
    total_dist_hydro = hor_dist_hydro + vert_dist_hydro
    hillshade_diff_morning_evening = hillshade_9am - hillshade_3pm

    # Build dictionary with training column names
    feature_dict = {
        "Elevation": elevation,
        "Slope": slope,
        "Aspect": aspect,
        "Horizontal_Distance_To_Hydrology": hor_dist_hydro,
        "Vertical_Distance_To_Hydrology": vert_dist_hydro,
        "Horizontal_Distance_To_Roadways": hor_dist_road,
        "Hillshade_9am": hillshade_9am,
        "Hillshade_Noon": hillshade_noon,
        "Hillshade_3pm": hillshade_3pm,
        "Horizontal_Distance_To_Fire_Points": hor_dist_fire,
        "Total_Distance_To_Hydrology": total_dist_hydro,
        "Hillshade_Diff_Morning_Evening": hillshade_diff_morning_evening,
    }

    # Add wilderness one-hot with correct names
    for i, val in enumerate(wilderness_dict[wilderness]):
        feature_dict[f"Wilderness_Area_{i+1}"] = val

    # Add soil one-hot with correct names
    for i, val in enumerate(soil_vector):
        feature_dict[f"Soil_Type_{i+1}"] = val

    # Convert to DataFrame
    input_df = pd.DataFrame([feature_dict])
    return input_df

# ✅ Session state to control prediction
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

if st.button("Predict Forest Cover Type"):
    st.session_state.predict_clicked = True

# ✅ Run prediction only when button is clicked
if st.session_state.predict_clicked:
    input_df = user_input_features()

    # Debug: show selected features vs input columns
    st.write("Selected features from training:", selected_features)
    st.write("Input DataFrame columns:", input_df.columns.tolist())

    # Reduce to selected features
    try:
        input_selected = input_df[selected_features]
        prediction = model.predict(input_selected)
        cover_type = label_encoder.inverse_transform(prediction)[0]
        st.success(f"🌳 Predicted Forest Cover Type: {cover_type}")
    except KeyError as e:
        st.error(f"Missing input features: {e}")

# ✅ Extra: Upload CSV to test predictions vs actual labels
st.subheader("📂 Test Model Accuracy with a CSV")
uploaded_file = st.file_uploader("Upload a CSV with true labels (must include 'Cover_Type' column)", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    # Ensure selected features exist
    missing = [f for f in selected_features if f not in test_df.columns]
    if missing:
        st.error(f"CSV is missing required features: {missing}")
    else:
        X_test = test_df[selected_features]
        y_true = test_df["Cover_Type"]
        y_pred = model.predict(X_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)

        # ✅ Decode actual labels if needed
        if pd.api.types.is_integer_dtype(y_true) or pd.api.types.is_numeric_dtype(y_true):
            y_true_labels = label_encoder.inverse_transform(y_true)
        else:
            y_true_labels = y_true

        results = pd.DataFrame({"Actual": y_true_labels, "Predicted": y_pred_labels})
        st.write(results.head(20))

        accuracy = (results["Actual"] == results["Predicted"]).mean()
        st.write(f"✅ Accuracy on uploaded test set: {accuracy:.2%}")