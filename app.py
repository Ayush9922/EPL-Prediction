import streamlit as st
import pandas as pd
import joblib
import numpy as np



# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="🏆 EPL Prediction Hub | AI Football Analytics",
    page_icon="https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Custom CSS
# --------------------------
st.markdown("""
<style>
    .stApp {
        background: url('https://images.unsplash.com/photo-1549921296-3d7c62fc7485?auto=format&fit=crop&w=1950&q=80') no-repeat center center fixed;
        background-size: cover;
        color: black;
    }
    h1, h2, h3 {
        color: black !important;
        font-weight: 800 !important;
    }
    section[data-testid="stSidebar"] {
        background: rgba(34, 40, 49, 0.95);
        color: white;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: white !important;
        border: 1.5px solid #00ADB5 !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        color: black;
    }
    .stButton > button {
        background: #00ADB5; 
        color: white; 
        border-radius: 12px; 
        padding: 12px 28px; 
        font-size: 16px; 
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #007f84;
        transform: translateY(-3px);
        box-shadow: 0px 5px 15px rgba(0,0,0,0.3);
    }
    .prediction-card {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.2);
        margin-bottom: 25px;
    }
    .header-image {
        border-radius: 20px;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.3);
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Load Models
# --------------------------
@st.cache_resource
def load_models():
    winner_model = joblib.load("match_winner_model.pkl")
    assists_model = joblib.load("top_assists_model.pkl")
    return winner_model, assists_model

winner_model, assists_model = load_models()

# --------------------------
# Sidebar Navigation
# --------------------------
st.sidebar.title("⚽ EPL Prediction Hub")
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

page = st.sidebar.radio("Navigate", ["🏠 Home", "🏆 Match Winner Predictor", "🎯 Player Assists Predictor"])
st.session_state.page = page

# --------------------------
# Team & Result Mappings
# --------------------------
team_mapping = {
    0: "Arsenal", 1: "Aston Villa", 2: "Bournemouth", 3: "Brentford", 4: "Brighton",
    5: "Chelsea", 6: "Crystal Palace", 7: "Everton", 8: "Fulham", 9: "Leeds United",
    10: "Leicester City", 11: "Liverpool", 12: "Man City", 13: "Man United", 14: "Newcastle",
    15: "Nottingham Forest", 16: "Southampton", 17: "Tottenham", 18: "West Ham", 19: "Wolves"
}
team_reverse_mapping = {v: k for k, v in team_mapping.items()}
half_time_result_mapping = {"Draw": 0, "Home Win": 1, "Away Win": 2}

# --------------------------
# Home Page - Professional Version
# --------------------------
if st.session_state.page == "🏠 Home":
    # Solid dark background
    st.markdown("""
    <style>
        .stApp {
            background-color: #1c1c1c;
            color: white;
        }
        h1 {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        p.subtitle {
            text-align: center;
            font-size: 1.2rem;
            opacity: 0.9;
            margin-top: 0;
            margin-bottom: 2rem;
        }
        .card {
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0px 6px 20px rgba(0,0,0,0.4);
            text-align: center;
            color: white;
            transition: transform 0.3s ease;
            margin-bottom: 20px;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card h2 {
            margin-bottom: 10px;
        }
        .card p {
            opacity: 0.9;
            font-size: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    
    st.markdown("""
    <div style='
        background-color: #008080; 
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    '>
        <h1 style='color: white !important; margin: 0;'>⚽ EPL Prediction Hub</h1>
    </div>
""", unsafe_allow_html=True)

    st.markdown("<p class='subtitle'>Select a predictor from the sidebar or click a card below</p>", unsafe_allow_html=True)

    # Two-column cards
    col1, col2 = st.columns(2, gap="large")

    # Match Winner Predictor Card
    with col1:
        st.markdown("""
        <div class='card' style='background-color: #11998e;'>
            <h2>🏆 Match Winner Predictor</h2>
            <p>Predict which team will win a football match using real match statistics and data-driven models.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Match Winner Predictor", key="btn_winner"):
            st.session_state.page = "🏆 Match Winner Predictor"
            st.experimental_set_query_params(page="winner")
            st.rerun()

    # Player Assists Predictor Card
    with col2:
        st.markdown("""
        <div class='card' style='background-color: #11998e;'>
            <h2>🎯 Player Assists Predictor</h2>
            <p>Estimate how many assists a player will make this season based on performance metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Player Assists Predictor", key="btn_assists"):
            st.session_state.page = "🎯 Player Assists Predictor"
            st.rerun()



# --------------------------
# Match Winner Predictor
# --------------------------
elif st.session_state.page == "🏆 Match Winner Predictor":
    st.image(
   "fancy-crave-qowyMze7jqg-unsplash.jpg",
    use_container_width=True
)


   
    st.markdown("""
<h1 style='
    text-align:center;
    font-weight:bold;
    background: linear-gradient(90deg, #1E90FF, #FF1493);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.0);
'>
🏆 Football Match Winner Prediction Hub
</h1>
""", unsafe_allow_html=True)


    st.markdown("<div class='prediction-card'>Enter match statistics below:</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        HomeTeam_name = st.selectbox("Home Team", list(team_reverse_mapping.keys()))
        HalfTimeHomeGoals = st.number_input("Half-Time Home Goals", min_value=0)
        HomeShots = st.number_input("Home Shots", min_value=0)
        HomeShotsOnTarget = st.number_input("Home Shots on Target", min_value=0)
        HomeCorners = st.number_input("Home Corners", min_value=0)
        HomeFouls = st.number_input("Home Fouls", min_value=0)
        HomeRedCards = st.number_input("Home Red Cards", min_value=0)
        HomeYellowCards = st.number_input("Home Yellow Cards", min_value=0)
    with col2:
        AwayTeam_name = st.selectbox("Away Team", list(team_reverse_mapping.keys()))
        HalfTimeAwayGoals = st.number_input("Half-Time Away Goals", min_value=0)
        HalfTimeResult_name = st.selectbox("Half-Time Result", list(half_time_result_mapping.keys()))
        AwayShots = st.number_input("Away Shots", min_value=0)
        AwayShotsOnTarget = st.number_input("Away Shots on Target", min_value=0)
        AwayCorners = st.number_input("Away Corners", min_value=0)
        AwayFouls = st.number_input("Away Fouls", min_value=0)
        AwayRedCards = st.number_input("Away Red Cards", min_value=0)
        AwayYellowCards = st.number_input("Away Yellow Cards", min_value=0)

    if st.button("⚡ Predict Match Winner"):
        try:
            input_df = pd.DataFrame([{
                'HomeTeam': team_reverse_mapping[HomeTeam_name],
                'AwayTeam': team_reverse_mapping[AwayTeam_name],
                'HalfTimeHomeGoals': HalfTimeHomeGoals,
                'HalfTimeAwayGoals': HalfTimeAwayGoals,
                'HalfTimeResult': half_time_result_mapping[HalfTimeResult_name],
                'HomeShots': HomeShots,
                'AwayShots': AwayShots,
                'HomeShotsOnTarget': HomeShotsOnTarget,
                'AwayShotsOnTarget': AwayShotsOnTarget,
                'HomeCorners': HomeCorners,
                'AwayCorners': AwayCorners,
                'HomeFouls': HomeFouls,
                'AwayFouls': AwayFouls,
                'HomeRedCards': HomeRedCards,
                'AwayRedCards': AwayRedCards,
                'HomeYellowCards': HomeYellowCards,
                'AwayYellowCards': AwayYellowCards
            }])
            input_df = input_df[winner_model.feature_names_in_]
            prediction = winner_model.predict(input_df)[0]
            prediction_mapping = {0: "Away Win",1: "Draw", 2: "Home Win"}
            winner_name = HomeTeam_name if prediction == 2 else AwayTeam_name if prediction == 0 else "Draw"
            st.success(f"🏁 Predicted Match Winner: **{winner_name} ({prediction_mapping[prediction]})**")
        except Exception as e:
            st.error(f"Error predicting match winner: {e}")

# --------------------------
# Player Assists Predictor
# --------------------------
elif st.session_state.page == "🎯 Player Assists Predictor":
    st.image(
        "66bf8b032fb651723828995.jpg",
        use_container_width=True
    )
    st.markdown("""
    <style>
        h1 {
            color: white !important;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)
    st.markdown("<h1 style='color:white;'>🎯 Player Assists Predictor</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        /* Make all Markdown headers white */
        h1, h2, h3, h4, h5, h6 {
            color: white !important;
        }

        /* Optional: make paragraph and label text white too */
        p, label, .stMarkdown {
            color: white !important;
        }

        /* Optional: darken input box backgrounds for contrast */
        .stTextInput, .stNumberInput, .stSelectbox {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: white !important;
        }

        /* Center your top title if you want consistency */
        h1 {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)
    
    with st.form("player_assists_form"):
        st.markdown("### 👤 Player Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 16, 40, 25)
        with col2:
            position = st.selectbox("Position", ["Forward", "Midfielder", "Defender", "Goalkeeper"])
            position_num = {"Forward": 1, "Midfielder": 2, "Defender": 3, "Goalkeeper": 4}[position]
        with col3:
            minutes_played = st.number_input("Minutes Played", 0, 4000, 2000)

        st.markdown("### 📊 Previous Season Stats")
        col4, col5, col6 = st.columns(3)
        with col4:
            prev_assists = st.number_input("Previous Season Assists", 0, 20, 5)
        with col5:
            goals_prev = st.number_input("Previous Season Goals", 0, 50, 5)
        with col6:
            key_passes = st.number_input("Key Passes", 0, 200, 50)

        st.markdown("### 🎯 Advanced Metrics")
        col7, col8, col9 = st.columns(3)
        with col7:
            expected_assists = st.number_input("Expected Assists (xA)", 0.0, 20.0, 5.0, 0.1)
        with col8:
            dribbles_completed = st.number_input("Dribbles Completed", 0, 200, 40)
        with col9:
            shots_assisted = st.number_input("Shots Assisted", 0, 200, 20)

        st.markdown("### 🏟️ Team Context")
        col10, col11, col12, col13 = st.columns(4)
        with col10:
           set_piece_involvement_label = st.selectbox("Set Piece Involvement?", ["No", "Yes"])
           set_piece_involvement = 1 if set_piece_involvement_label == "Yes" else 0
        with col11:
            club_total_goals = st.number_input("Club Total Goals", 0, 150, 60)
        with col12:
            club_league_rank = st.number_input("Club League Rank", 1, 20, 5)
        with col13:
            big6_club_label = st.selectbox("Big 6 Club?", ["No", "Yes"])
            big6_club_feature = 1 if big6_club_label == "Yes" else 0

        st.markdown("### 📈 Per 90 & Ratio Stats")
        col14, col15, col16 = st.columns(3)
        with col14:
            club_attack_share = st.number_input("Club Attack Share", 0.0, 1.0, 0.35, 0.01)
        with col15:
            club_xg = st.number_input("Club xG", 0.0, 100.0, 50.0, 0.1)
        with col16:
            assists_per_90 = st.number_input("Assists per 90", 0.0, 2.0, 0.25, 0.01)

        col17, col18, col19, col20 = st.columns(4)
        with col17:
            goals_per_90 = st.number_input("Goals per 90", 0.0, 2.0, 0.15, 0.01)
        with col18:
            contribution_ratio = st.number_input("Contribution Ratio", 0.0, 5.0, 0.35, 0.01)
        with col19:
            dribbles_per_90 = st.number_input("Dribbles per 90", 0.0, 10.0, 2.5, 0.1)
        with col20:
            shots_assisted_per_90 = st.number_input("Shots Assisted per 90", 0.0, 10.0, 2.2, 0.1)

        submitted = st.form_submit_button("🔮 Predict Assists")
        
        if submitted:
            if assists_model:
                with st.spinner("🔄 Analyzing player data..."):
                    input_values = [
                        age, position_num, minutes_played, prev_assists, goals_prev,
                        key_passes, expected_assists, dribbles_completed, shots_assisted,
                        set_piece_involvement, club_total_goals, club_league_rank, big6_club_feature,
                        club_attack_share, club_xg, assists_per_90, goals_per_90,
                        contribution_ratio, dribbles_per_90, shots_assisted_per_90
                    ]

                    input_cols = [
                        'Age', 'Position', 'Minutes_Played', 'Assists_prev_season',
                        'Goals_prev_season', 'Key_Passes', 'Expected_Assists_(xA)',
                        'Dribbles_Completed', 'Shots_Assisted', 'Set_Piece_Involvement',
                        'Club_Total_Goals', 'Club_League_Rank', 'Big6_Club_Feature',
                        'Club_Attack_Share', 'Club_xG', 'Assists_per_90', 'Goals_per_90',
                        'Contribution_Ratio', 'Dribbles_per_90', 'Shots_Assisted_per_90'
                    ]

                    input_df = pd.DataFrame([input_values], columns=input_cols)

                    # Apply preprocessing if needed
                    if "poly" in globals():
                        input_df = poly.transform(input_df)
                    if "scaler" in globals():
                        input_df = scaler.transform(input_df)

                    # Prediction
                    prediction = float(assists_model.predict(input_df)[0])

                    # Result display
                    st.markdown("---")
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
    <h1 style='color: white; margin: 0;'>🎯 Predicted Assists</h1>
    <h2 style='color: white; margin-top: 1rem; font-size: 3rem;'>{prediction:.1f}</h2>
    <p style='color: white; opacity: 0.9;'>Expected assists for this season</p>
</div>


                    """, unsafe_allow_html=True)

                    # Performance metrics
                    col_p1, col_p2, col_p3 = st.columns(3)
                    with col_p1:
                        st.metric("Performance Level", 
                                  "Elite" if prediction >= 12 else "Good" if prediction >= 8 else "Average")
                    with col_p2:
                        st.metric("Rank Potential", 
                                  "Top 5" if prediction >= 12 else "Top 10" if prediction >= 8 else "Top 20")
                    with col_p3:
                        confidence = "High" if prev_assists >= 5 else "Medium"
                        st.metric("Prediction Confidence", confidence)

                    # Player insights
                    st.markdown("#### 💡 Player Analysis")
                    if prediction > prev_assists * 1.2:
                        st.success("📈 Significant improvement expected!")
                    elif prediction > prev_assists:
                        st.info("📊 Moderate improvement projected")
                    else:
                        st.warning("⚠️ Performance may decline or remain stable")
                    
                    if big6_club_feature == 1:
                        st.markdown("- ⭐ Playing for a top club provides better opportunities")
                    if minutes_played >= 2500:
                        st.markdown("- ✅ High playing time ensures consistent contribution")
                    if expected_assists > prev_assists:
                        st.markdown("- 🎯 Underperformed xA last season – regression to mean expected")
# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #bbb; padding: 2rem; font-size: 0.9rem;'>
        <p style='animation: fadeIn 2s ease-in;'>🚀 Developed with Streamlit by <b>Ayush</b></p>
        <style>
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
        </style>
    </div>
""", unsafe_allow_html=True)

