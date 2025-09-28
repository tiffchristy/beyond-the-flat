import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
rmse = pd.read_csv('model_metrics.csv')

# --- Helper Functions ---
def get_user_input(profile_choice):
    """
    Retrieves user input for different profiles or provides a free-style form.
    Args:
        profile_choice (str): The selected profile.
    Returns:
        dict: A dictionary of user-defined features.
    """
    hardcoded_profiles = {
        'Kishan': {
            "floor_area_sqm": 94,
            "Tranc_Year": 2021,
            "hdb_age": 16,
            "max_floor_lvl": 30,
            "demand": 40,
            "amenity_proximity_score": 555,
            "transport_proximity_score": 666,
            "amenities_within_1km": 3,
            "dist_nearest_top_school": 787,
            "mrt_development": 3,
            "affluent_index": 200,
            "avg_subs_sch": 1.5,
            "num_top_sch": 2,
        },
        'Priscilla': {
            "floor_area_sqm": 110,
            "Tranc_Year": 2015,
            "hdb_age": 24,
            "max_floor_lvl": 10,
            "demand": 88,
            "amenity_proximity_score": 676,
            "transport_proximity_score": 600,
            "amenities_within_1km": 4,
            "dist_nearest_top_school": 890,
            "mrt_development": 4,
            "affluent_index": 232,
            "avg_subs_sch": 1.0,
            "num_top_sch": 0,
        }
    }

    if profile_choice in hardcoded_profiles:
        with top_container:
            st.subheader("Features")
            col_1, col_2, col_3 = st.columns(3)
            # Display hardcoded values as read-only sliders/number inputs
            Month = col_1.selectbox("Month", [1,2,3,4,5,6,7,8,9,10,11,12])
            floor_area_sqm = col_1.slider("Floor Area (sqm)", min_value=0, max_value=300, value=hardcoded_profiles[profile_choice]["floor_area_sqm"])
            Tranc_Year = col_1.slider("Transaction Year", min_value=2000, max_value=2030, value=hardcoded_profiles[profile_choice]["Tranc_Year"])
            hdb_age = col_1.number_input("HDB Age", min_value=0, max_value=100, value=hardcoded_profiles[profile_choice]["hdb_age"])
            max_floor_lvl = col_2.number_input("Max Floor Level", min_value=1, max_value=50, value=hardcoded_profiles[profile_choice]["max_floor_lvl"])
            demand = col_2.number_input("No. of Units sold", min_value=0, max_value=130, value=hardcoded_profiles[profile_choice]["demand"])
            amenity_proximity_score = col_2.slider("Dist. to nearest Mall/Hawker(m)", min_value=0, max_value=5000, value=hardcoded_profiles[profile_choice]["amenity_proximity_score"])
            transport_proximity_score = col_2.slider("Dist. to nearest MRT/Bus(m)", min_value=0, max_value=5000, value=hardcoded_profiles[profile_choice]["transport_proximity_score"])
            amenities_within_1km = col_3.number_input("How many amenities do you want within 1km (mall/hawker)", min_value=0, max_value=17, value=hardcoded_profiles[profile_choice]["amenities_within_1km"])
            dist_nearest_top_school = col_3.slider("Distance to Nearest Top School (m)", min_value=30, max_value=7000, value=hardcoded_profiles[profile_choice]["dist_nearest_top_school"])
            mrt_development = col_3.number_input("Any New MRT coming up nearby?", min_value=1, max_value=5, value=hardcoded_profiles[profile_choice]["mrt_development"])
            affluent_index = col_3.number_input("Affluent Index", min_value=0, max_value=525, value=hardcoded_profiles[profile_choice]["affluent_index"])
            avg_subs_sch = col_1.number_input("Primary school subscription", min_value=0.0, max_value=2.0, value=hardcoded_profiles[profile_choice]["avg_subs_sch"])
            num_top_sch = col_2.slider("Number of top schools in the area", min_value=0, max_value=4, value=hardcoded_profiles[profile_choice]["num_top_sch"])
            
            return {
                "Month": Month,
                "floor_area_sqm": floor_area_sqm,
                "Tranc_Year": Tranc_Year,
                "hdb_age": hdb_age,
                "max_floor_lvl": max_floor_lvl,
                "demand": demand,
                "amenity_proximity_score": amenity_proximity_score,
                "transport_proximity_score": transport_proximity_score,
                "amenities_within_1km": amenities_within_1km,
                "dist_nearest_top_school": dist_nearest_top_school,
                "mrt_development": mrt_development,
                "affluent_index": affluent_index,
                "avg_subs_sch": avg_subs_sch,
                "num_top_sch": num_top_sch,
            }

    else: # 'Free-style'
        with top_container:
            st.subheader("Features")
            col_1, col_2, col_3 = st.columns(3)
            Month = col_1.selectbox("Month", [1,2,3,4,5,6,7,8,9,10,11,12])
            floor_area_sqm = col_1.slider("Floor Area (sqm)", min_value=0, max_value=300, value=50)
            Tranc_Year = col_1.slider("Transaction Year", min_value=2012, max_value=2021, value=2015)
            hdb_age = col_1.number_input("HDB Age", min_value=0, max_value=100, value=30)
            max_floor_lvl = col_2.number_input("Max Floor Level", min_value=1, max_value=50, value=10)
            demand = col_2.number_input("No. of Units sold", min_value=0, max_value=130, value=0)
            amenity_proximity_score = col_2.slider("Dist. to nearest Mall/Hawker(m)", min_value=0, max_value=5000, value=1000)
            transport_proximity_score = col_2.slider("Dist. to nearest MRT/Bus(m)", min_value=0, max_value=5000, value=1000)
            amenities_within_1km = col_3.number_input("How many amenities do you want within 1km (mall/hawker)", min_value=0, max_value=17, value=1)
            dist_nearest_top_school = col_3.slider("Distance to Nearest Top School (m)", min_value=30, max_value=7000, value=500)
            mrt_development = col_3.number_input("Any New MRT coming up nearby?", min_value=1, max_value=5, value=2)
            affluent_index = col_3.number_input("Affluent Index", min_value=0, max_value=525, value=100)
            avg_subs_sch = col_1.number_input("Primary school subscription", min_value=0.0, max_value=2.0, value=1.0)
            num_top_sch = col_2.slider("Number of top schools in the area", min_value=0, max_value=4, value=0)
            
            return {
                "Month": Month,
                "floor_area_sqm": floor_area_sqm,
                "Tranc_Year": Tranc_Year,
                "hdb_age": hdb_age,
                "max_floor_lvl": max_floor_lvl,
                "demand": demand,
                "amenity_proximity_score": amenity_proximity_score,
                "transport_proximity_score": transport_proximity_score,
                "amenities_within_1km": amenities_within_1km,
                "dist_nearest_top_school": dist_nearest_top_school,
                "mrt_development": mrt_development,
                "affluent_index": affluent_index,
                "avg_subs_sch": avg_subs_sch,
                "num_top_sch": num_top_sch,
            }

def get_economic_values(df, user_data):
    """
    Calculates age-related and economic values based on user input.
    """
    Tranc_Year = user_data.get('Tranc_Year')
    Month = user_data.get('Month')
    hdb_age = user_data.get('hdb_age')
    
    lease_commence_date = 2021 - hdb_age
    calc_age_at_sale = Tranc_Year - lease_commence_date

    def get_MHI(df, year):
        year_mask = df['Tranc_Year'] == year
        filtered_df = df[year_mask]
        return filtered_df['MHI'].iloc[0] if not filtered_df.empty else None

    def get_cpi_and_gdpm_values(df, year, month):
        filtered_df = df[(df['Tranc_Year'] == year) & (df['Month'] == month)]
        if not filtered_df.empty:
            return filtered_df['CPI'].iloc[0], filtered_df['GDPM'].iloc[0]
        else:
            return None, None
    
    user_data["lease_commence_date"] = lease_commence_date
    user_data["age_at_sale"] = calc_age_at_sale
    user_data["CPI"] = get_cpi_and_gdpm_values(df, Tranc_Year, Month)[0]
    user_data["GDPM"] = get_cpi_and_gdpm_values(df, Tranc_Year, Month)[1]
    user_data["MHI"] = get_MHI(df, Tranc_Year)
    
    return user_data

def prepare_input(data, feature_list):
    """Prepares the input data for the model."""
    return pd.DataFrame([{feature: data.get(feature, 0) for feature in feature_list}])

def create_and_display_chart(profile_choice, df):
    """Creates and displays a Plotly chart based on the selected profile."""
    if profile_choice == 'Kishan':
        fig = px.scatter(
                        x=df['dist_nearest_top_school'],
                        y=df['resale_price'],
                        orientation='h',
                        title='Size:No.of top schools, Colour:Average school subscription',
                        size=df['num_top_sch'],
                        color=df['avg_subs_sch'],
                        color_continuous_scale='temps')
        fig.update_layout(xaxis_title="Nearest distance to top school",
                        yaxis_title="Resale Price",
                        template='plotly_white',
                        height=500)
        
    elif profile_choice == 'Priscilla':
        fig = px.histogram(df,
                        x='dist_nearest_top_school',
                        y='resale_price',
                        labels={'dist_nearest_top_school': 'Distance to nearest top school', 'resale_price': 'Resale Price'})

    elif profile_choice == 'Free-style':
        fig = px.scatter(
                    x=df['amenity_proximity_score'],
                    y=df['resale_price'],
                    orientation='h',
                    title='Size:Median Household Income, Colour:Demand',
                    size=df['MHI'],
                    color=df['demand'],
                    color_continuous_scale='sunsetdark',
                    opacity=0.7)
        fig.update_layout(xaxis_title="Amenities Distance",
                        yaxis_title="Resale Price",
                        template='plotly_white',
                        height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    


# --- App Configuration ---
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    /* Main app background color */
    .stApp {
        background-color: #00008B; /* Dark Blue */
        color: black;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #009688;
        color: white;
        font-size: 16px;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        border: 2px solid #009688;
    }
    div.stButton > button:hover {
        background-color: #00796b;
        border: 2px solid #00796b;
    }

    /* User inputs */
    .stTextInput > div > div > input, .stNumberInput > div > div > input {
        border: 2px solid #555555;
        border-radius: 5px;
        background-color: white;
        color: black;
    }

    /* Containers */
    [data-testid="stVerticalBlock"] {
        background-color: white !important;
        padding: 5px;
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Headers */
    .header-content {
        display: flex;
        align-items: center;
        gap: 20px;
        width: 100%;
    }
    .header-title {
        margin: 0;
        color: #333333;
        text-align: center;
        width: 100%;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: black;
    }

    [data-testid="column"] > div:nth-child(2) h1 {
        text-align: center !important;
        width: 100%;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Model ---
with open("model.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

# --- Containers ---
header_container = st.container(border=True)
top_container = st.container(border=True)
middle_container = st.container(border=True)
feature_imp_container = st.container(border=True)
bottom_container = st.container(border=True)
data_dict_container = st.container(border=True)

# --- Main App Logic ---
# Header and dropdown
with header_container:
    col1, col2 = st.columns([1, 4])
    with col1:
        logo = Image.open('pinnacle_logo.png')
        st.image(logo, use_container_width=True)
    with col2:
        st.markdown("<h1 style='text-align: center; class='header-title'>Beyond The Flat</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    profile_choice = st.selectbox(
        "Choose a prediction profile",
        ("Free-style", "Priscilla", "Kishan")
    )

# Middle container
with middle_container:
    col1, col2 = st.columns(2)
    with col1:
        st.header("Predict HDB Price")
        user_data_base = get_user_input(profile_choice)
        user_data = get_economic_values(df, user_data_base)
        
        features = [
            "demand", "age_at_sale", "max_floor_lvl", "lease_commence_date",
            "Month", "Tranc_Year", "affluent_index", "hdb_age", "floor_area_sqm",
            "amenity_proximity_score", "transport_proximity_score", "amenities_within_1km",
            "avg_subs_sch", "num_top_sch", "dist_nearest_top_school", "mrt_development",
            "CPI", "GDPM", "MHI",
        ]

        if 'predict_clicked' not in st.session_state:
            st.session_state.predict_clicked = False

        def set_predict_clicked():
            st.session_state.predict_clicked = True

        st.button("Predict", on_click=set_predict_clicked)
    
    with col2:
        if st.session_state.predict_clicked:
            if all(value is not None for value in user_data.values()):
                input_array = prepare_input(user_data, features)
                prediction = model_pipeline.predict(X=input_array)
                prediction_value = prediction[0]
                rmse = rmse['rmse'].iloc[0]
                lower_price_range = prediction_value - rmse
                upper_price_range = prediction_value + rmse
                st.subheader("Predicted HDB Resale Price")
                st.write(f"The PREDICTED resale price is: ${prediction_value:,.2f}.")
                st.write(f"The predicted price range is:\${lower_price_range:,.2f} - \${upper_price_range:,.2f}.")                
            else:
                st.error("Missing data to make a prediction. Please ensure the selected year and month exist in the dataset.")
                
# Feature importance chart
with feature_imp_container:
    if st.session_state.predict_clicked:
        lgb.plot_importance(model_pipeline, importance_type='gain', figsize=(10, 6))
        plt.title('LightGBM Feature Importance')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        st.pyplot(plt)
        st.write("The top 5 drivers of resale price are floor area, affluence, max floor level, primary school subscriptions and finally amenities within 1km.")
        st.write("Let's see how to use the interactive chart to represent a buyer or seller. ")

# Bottom container for charts
with bottom_container:
    st.subheader("Finding your target audience with the new features")
    if st.session_state.predict_clicked:
        create_and_display_chart(profile_choice, df)

# Data dictionary chart
with data_dict_container:
    st.subheader("Data Dictionary")
    # 1. Define the data dictionary
    data_dictionary = {
        "demand": "The number of units sold for a specific flat type.",
        "age_at_sale": "The age of the flat at the time of sale.",
        "max_floor_lvl": "The highest floor level of the HDB block.",
        "lease_commence_date": "The year the HDB flat's lease started.",
        "Month": "The month of the transaction.",
        "Tranc_Year": "The year of the transaction.",
        "affluent_index": "An index representing the affluence of the area.",
        "hdb_age": "The current age of the HDB flat.",
        "floor_area_sqm": "The floor area of the flat in square meters.",
        "amenity_proximity_score": "The distance to the nearest major amenity (e.g., mall, hawker center).",
        "transport_proximity_score": "The distance to the nearest transport node (e.g., MRT, bus interchange).",
        "amenities_within_1km": "The count of major amenities within a 1km radius.",
        "avg_subs_sch": "The average subscription rate of primary schools in the area.",
        "num_top_sch": "The number of top-ranking schools in the area.",
        "dist_nearest_top_school": "The distance to the closest top-ranking school.",
        "mrt_development": "A score indicating new MRT developments in the vicinity.",
        "CPI": "Consumer Price Index for the transaction period.",
        "GDPM": "Gross Domestic Product per capita for the transaction period.",
        "MHI": "Median Household Income for the transaction period.",
        }

    # The list of features for the selectbox
    features = list(data_dictionary.keys())

    # 2. Create the selectbox and get the user's choice
    selected_feature = st.selectbox("Select a feature to see its description", features)

    # 3. Display the description based on the selection
    if selected_feature:
        description = data_dictionary.get(selected_feature, "Description not found.")
        st.subheader(f"**Description for '{selected_feature}':**")
        st.info(description)
    