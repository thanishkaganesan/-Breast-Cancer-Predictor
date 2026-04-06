Logisitic Regression
LLM (Gemini 2.5 flash)
streamlit for frontend

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import google.generativeai as genai
import joblib
import os
import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Security Best Practice: Use Streamlit Secrets for API Keys ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "YOUR_DEFAULT_FALLBACK_KEY_IF_NEEDED")

# Page Configuration
st.set_page_config(
    page_title="AI-Powered Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- RETAINED: Original CSS from your script ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 2rem;
        border-radius: 10px; margin-bottom: 2rem; text-align: center; color: white;
    }
    .risk-low {
        background: linear-gradient(90deg, #00b09b, #96c93d); color: white; padding: 1rem;
        border-radius: 10px; text-align: center; font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(90deg, #f7971e, #ffd200); color: white; padding: 1rem;
        border-radius: 10px; text-align: center; font-weight: bold;
    }
    .risk-high {
        background: linear-gradient(90deg, #ff416c, #ff4b2b); color: white; padding: 1rem;
        border-radius: 10px; text-align: center; font-weight: bold;
    }
    .alert-info {
        background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460;
        padding: 0.75rem 1.25rem; margin-bottom: 1rem; border-radius: 0.25rem;
    }
    .stButton > button {
        background-color: #2a5298; color: white; border-radius: 10px; border: none;
        padding: 0.5rem 1rem; font-weight: bold; transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1e3c72; color: white; transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# --- Feature Definitions ---
FEATURE_GROUPS = {
    'Mean Values': {
        'features': ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'],
        'defaults': [14.0, 19.0, 92.0, 655.0, 0.096, 0.104, 0.089, 0.048, 0.181, 0.063],
    },
    'Standard Error': {
        'features': ['radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se'],
        'defaults': [0.405, 1.22, 2.87, 40.3, 0.007, 0.025, 0.032, 0.012, 0.021, 0.004],
    },
    'Worst Values': {
        'features': ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'],
        'defaults': [16.3, 25.7, 107.0, 880.0, 0.132, 0.254, 0.272, 0.115, 0.290, 0.084],
    }
}
FEATURE_NAMES = [feature for group in FEATURE_GROUPS.values() for feature in group['features']]
FEATURE_DEFAULTS = [default for group in FEATURE_GROUPS.values() for default in group['defaults']]

# --- Session State Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = "Description"
if 'current_inputs' not in st.session_state:
    st.session_state.current_inputs = FEATURE_DEFAULTS.copy()
if 'model' not in st.session_state:
    st.session_state.model = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False

# --- Helper Functions ---
@st.cache_resource
def load_hardcoded_model():
    model_path = 'logistic_regression_model.joblib'
    try:
        if os.path.exists(model_path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return joblib.load(model_path)
        return None
    except Exception:
        return None

@st.cache_data
def generate_ai_recommendations(risk_level, probability, top_features):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.3)
        prompt = ChatPromptTemplate.from_template("""
        As a medical AI assistant, provide concise, practical recommendations for a breast cancer risk assessment.
        Risk: {risk_level} ({probability}). Key Factors: {top_features}.
        Provide: 1. Next steps. 2. Lifestyle tips (diet). 3. Follow-up advice. 4. When to see a doctor.
        Instructions: Be brief. Use simple language. Emphasize this is not a diagnosis. No markdown.
        """)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "risk_level": risk_level, "probability": f"{probability:.1%}", "top_features": ', '.join(top_features)
        })
    except Exception as e:
        return f"Error generating AI recommendations: {str(e)}"

def load_sample_data(sample_type):
    samples = {
        'high_risk': [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
        'low_risk': [11.42, 20.38, 77.58, 386.1, 0.1425, 0.2839, 0.2414, 0.1052, 0.2597, 0.09744, 0.4956, 1.156, 3.445, 27.23, 0.00911, 0.07458, 0.05661, 0.01867, 0.05963, 0.009208, 14.91, 26.5, 98.87, 567.7, 0.2098, 0.8663, 0.6869, 0.2575, 0.6638, 0.173],
        'defaults': FEATURE_DEFAULTS
    }
    return samples.get(sample_type, FEATURE_DEFAULTS)

# --- Startup Logic ---
st.session_state.model = load_hardcoded_model()
try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_DEFAULT_FALLBACK_KEY_IF_NEEDED":
        genai.configure(api_key=GEMINI_API_KEY)
        st.session_state.api_configured = True
except Exception:
    st.session_state.api_configured = False


# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://via.placeholder.com/200x100/1e3c72/ffffff?text=Medical+AI", width=200)
    st.title("Navigation")
    
    # Define pages and their icons
    pages = {
        "Description": "📖",
        "Patient Input": "📝",
        "Prediction & Visualization": "📊",
        "AI Recommendations": "🤖",
        "History & Reporting": "📋"
    }
    
    for page_name, icon in pages.items():
        if st.button(f"{icon} {page_name}", use_container_width=True):
            st.session_state.page = page_name

    st.divider()
    st.title("🔧 System Status")
    
    # Model Status
    if st.session_state.model:
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ Model loading failed!")

    # API Status
    if st.session_state.api_configured:
        st.success("✅ Gemini API configured!")
    else:
        st.error("❌ API configuration failed!")


# --- Page Routing ---
if st.session_state.model is None:
    st.error("### 🚨 Critical Error: Model Not Found")
    st.markdown("The application cannot start because the logistic_regression_model.joblib file could not be loaded. Please ensure the model file is in the same directory as the script and restart.")
    st.stop()

# --- 1. Description Page ---
if st.session_state.page == "Description":
    st.markdown("<div class='main-header'><h1>AI-Powered Disease Prediction System</h1></div>", unsafe_allow_html=True)
    st.markdown("### Welcome to the Advanced Breast Cancer Risk Assessment Tool")
    st.markdown("""
    This application leverages a machine learning model to predict the probability of breast cancer malignancy based on cellular characteristics derived from digitized images of a fine-needle aspirate (FNA) of a breast mass.

    #### How It Works:
    1.  *Navigate* using the sidebar to move between sections.
    2.  *Patient Input*: Go to the input page to enter the 30 required cellular measurements. You can enter them manually, load sample data, or upload a CSV file.
    3.  *Prediction*: Once the data is entered, the "Predict Risk" button on the input page will run the model.
    4.  *View Results*: The results, including a risk score, visualizations, and AI-powered recommendations, will be available on their respective pages after a prediction is made.
    5.  *History*: All predictions are logged on the History & Reporting page for review and export.

    *Disclaimer: This tool is for educational and informational purposes only. It is **not a substitute* for professional medical advice, diagnosis, or treatment.
    """)
    st.info("To begin, select *📝 Patient Input* from the sidebar.", icon="👈")

# --- 2. Patient Input Page ---
elif st.session_state.page == "Patient Input":
    st.markdown("<div class='main-header'><h1>📝 Patient Data Input</h1></div>", unsafe_allow_html=True)
    
    # --- Data Loading options ---
    st.subheader("⚡ Quick Fill & Upload Options")
    col1, col2, col3 = st.columns(3)
    if col1.button("🔄 Reset to Defaults", use_container_width=True):
        st.session_state.current_inputs = load_sample_data('defaults')
        st.rerun()
    if col2.button("📋 High Risk Sample", use_container_width=True):
        st.session_state.current_inputs = load_sample_data('high_risk')
        st.rerun()
    if col3.button("🟢 Low Risk Sample", use_container_width=True):
        st.session_state.current_inputs = load_sample_data('low_risk')
        st.rerun()
        
    patient_file = st.file_uploader("Upload patient data CSV (optional)", type=['csv'])
    if patient_file:
        try:
            patient_data = pd.read_csv(patient_file)
            st.session_state.current_inputs = patient_data.iloc[0, :30].astype(float).tolist()
            st.success("CSV data loaded. You can now proceed to predict.")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")

    # --- Input Form ---
    with st.expander("Enter Patient Data (All 30 Features)", expanded=True):
        for group_name, group_data in FEATURE_GROUPS.items():
            st.markdown(f"#### {group_name}")
            cols = st.columns(4)
            for i, feature in enumerate(group_data['features']):
                with cols[i % 4]:
                    feature_idx = FEATURE_NAMES.index(feature)
                    value = st.number_input(
                        feature.replace('_', ' ').title(),
                        value=float(st.session_state.current_inputs[feature_idx]),
                        step=0.001, format="%.6f", key=f"feature_{feature_idx}"
                    )
                    st.session_state.current_inputs[feature_idx] = value
    
    st.divider()
    if st.button("🔬 Predict Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing patient data..."):
            feature_values = list(st.session_state.current_inputs)
            feature_data = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
            
            # Make prediction
            probability = st.session_state.model.predict_proba(feature_data)[0][1]
            prediction = st.session_state.model.predict(feature_data)[0]
            
            # Store results in session state
            st.session_state.prediction_result = {
                "probability": probability, "prediction": prediction, "inputs": feature_values
            }
            
            # Add to history
            if probability < 0.33: risk_level = "Low"
            elif probability < 0.66: risk_level = "Medium"
            else: risk_level = "High"

            st.session_state.predictions_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'risk_level': risk_level,
                'probability': f"{probability:.3f}",
                'prediction': 'Malignant' if prediction == 1 else 'Benign'
            })

        st.success("Prediction complete! Navigate to the other pages to see the results.")
        st.session_state.page = "Prediction & Visualization"
        st.rerun()

# --- 3. Prediction & Visualization Page ---
elif st.session_state.page == "Prediction & Visualization":
    st.markdown("<div class='main-header'><h1>📊 Prediction & Visualization</h1></div>", unsafe_allow_html=True)
    
    if not st.session_state.prediction_result:
        st.info("Please enter patient data and run a prediction on the 'Patient Input' page first.", icon="👈")
    else:
        result = st.session_state.prediction_result
        probability = result['probability']
        prediction = result['prediction']
        
        if probability < 0.33: risk_level, risk_class, risk_color = "Low", "risk-low", "#00b09b"
        elif probability < 0.66: risk_level, risk_class, risk_color = "Medium", "risk-medium", "#f7971e"
        else: risk_level, risk_class, risk_color = "High", "risk-high", "#ff416c"
            
        st.subheader("Risk Assessment Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"<div class='{risk_class}'><h3>Risk Level: {risk_level}</h3><p>Probability of Malignancy: {probability:.1%}</p><p>Classification: {'Malignant' if prediction == 1 else 'Benign'}</p></div>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=probability * 100, title={'text': "Risk Gauge"},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': risk_color}}
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.subheader("Top Contributing Features")
            if hasattr(st.session_state.model, 'coef_'):
                coef = st.session_state.model.coef_[0]
                feature_importance = pd.DataFrame({'Feature': FEATURE_NAMES, 'Contribution': coef * result['inputs']})
                feature_importance = feature_importance.sort_values('Contribution', key=abs, ascending=False).head(10)
                
                fig_bar = px.bar(feature_importance.sort_values('Contribution', ascending=True),
                                x='Contribution', y='Feature', orientation='h', color='Contribution',
                                color_continuous_scale='RdBu_r', title="Feature Contribution to Prediction")
                fig_bar.update_layout(height=400, margin=dict(l=150))
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Feature importance analysis not available for this model type.")

# --- 4. AI Recommendations Page ---
elif st.session_state.page == "AI Recommendations":
    st.markdown("<div class='main-header'><h1>🤖 AI-Powered Recommendations</h1></div>", unsafe_allow_html=True)

    if not st.session_state.prediction_result:
        st.info("Please run a prediction on the 'Patient Input' page to generate recommendations.", icon="👈")
    elif not st.session_state.api_configured:
        st.warning("AI Recommendations are unavailable. The Gemini API is not configured.", icon="⚠️")
    else:
        with st.spinner("Generating personalized recommendations..."):
            result = st.session_state.prediction_result
            probability = result['probability']
            
            if probability < 0.33: risk_level = "Low"
            elif probability < 0.66: risk_level = "Medium"
            else: risk_level = "High"

            top_features = []
            if hasattr(st.session_state.model, 'coef_'):
                coef = st.session_state.model.coef_[0]
                feature_importance = pd.DataFrame({'Feature': FEATURE_NAMES, 'Contribution': coef * result['inputs']})
                top_features = feature_importance.sort_values('Contribution', key=abs, ascending=False).head(5)['Feature'].tolist()

            recommendations = generate_ai_recommendations(risk_level, probability, top_features)
            
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #1e3c72, #2a5298); padding: 25px; border-radius: 12px; color: white; margin: 20px 0;">
                <h4 style="margin-top:0;">Personalized Medical Recommendations</h4>
                <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 15px; margin-top: 15px;">
                    {recommendations.replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.warning("This is an AI-generated summary and not a substitute for professional medical advice.", icon="⚠️")

# --- 5. History & Reporting Page ---
elif st.session_state.page == "History & Reporting":
    st.markdown("<div class='main-header'><h1>📋 Prediction History & Reporting</h1></div>", unsafe_allow_html=True)

    if not st.session_state.predictions_history:
        st.info("No predictions have been made in this session yet.", icon="ℹ️")
    else:
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        col1, col2 = st.columns([1, 1])
        if col1.button("🗑️ Clear History", use_container_width=True):
            st.session_state.predictions_history = []
            st.session_state.prediction_result = None
            st.rerun()

        csv = history_df.to_csv(index=False).encode('utf-8')
        col2.download_button(
             label="📥 Export to CSV", data=csv,
             file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
             mime="text/csv", use_container_width=True
        )
        
        st.dataframe(history_df, use_container_width=True)
        
        st.subheader("History Overview")
        risk_counts = history_df['risk_level'].value_counts()
        fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index, title="Risk Level Distribution",
                        color_discrete_map={'Low': '#00b09b', 'Medium': '#f7971e', 'High': '#ff416c'})
        st.plotly_chart(fig_pie, use_container_width=True)
