import joblib
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from typing import Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
FEATURE_COLUMNS = [
    'AgeCategory', 'Sex', 'BMI', 'PhysicalHealth', 'MentalHealth', 'Smoking',
    'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Race', 'Diabetic',
    'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer'
]

RISK_THRESHOLDS = {
    'low': 25,
    'moderate': 50,
    'high': 75
}

# Enhanced CSS with animations and better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeIn 1s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .sub-header {
        font-size: 1.5rem;
        color: #4B77BE;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E8F4FD;
        padding-bottom: 0.5rem;
    }

    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }

    .result-box:hover {
        transform: translateY(-2px);
    }

    .high-risk {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 5px solid #F44336;
    }

    .moderate-risk {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border-left: 5px solid #FF9800;
    }

    .low-risk {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 5px solid #4CAF50;
    }

    .info-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #2196F3;
    }

    .stButton>button {
        background: linear-gradient(135deg, #4B77BE 0%, #3B5998 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(75, 119, 190, 0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #3B5998 0%, #2A4480 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(75, 119, 190, 0.4);
    }

    .form-section {
        background: linear-gradient(135deg, #F8F9FA 0%, #FFFFFF 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #E9ECEF;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem 0;
    }

    .risk-gauge {
        text-align: center;
        margin: 1rem 0;
    }

    .disclaimer {
        background: #FFF9C4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFC107;
        margin: 1rem 0;
    }

    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
</style>
""", unsafe_allow_html=True)


class HeartDiseasePredictor:
    """Heart Disease Risk Prediction Model Handler"""

    def __init__(self):
        self.model = None
        self.model_loaded = False

    @st.cache_resource
    def load_model(_self):
        """Load the trained model with error handling"""
        try:
            model = joblib.load('health_pipeline.joblib')
            _self.model_loaded = True
            logger.info("Model loaded successfully")
            return model
        except FileNotFoundError:
            logger.error("Model file not found")
            st.error("⚠️ Model file not found. Please ensure 'health_pipeline.joblib' is in the correct directory.")
            return None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            st.error(f"⚠️ Error loading model: {str(e)}")
            return None

    def validate_input(self, input_data: pd.DataFrame) -> bool:
        """Validate input data"""
        try:
            # Check if all required columns are present
            missing_cols = set(FEATURE_COLUMNS) - set(input_data.columns)
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                return False

            # Check for valid ranges
            if input_data['BMI'].iloc[0] < 10 or input_data['BMI'].iloc[0] > 100:
                st.error("BMI must be between 10 and 100")
                return False

            if input_data['SleepTime'].iloc[0] < 1 or input_data['SleepTime'].iloc[0] > 24:
                st.error("Sleep time must be between 1 and 24 hours")
                return False

            return True
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            st.error(f"Input validation error: {str(e)}")
            return False

    def predict(self, input_data: pd.DataFrame) -> Tuple[str, float]:
        """Make prediction with error handling"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")

            # Validate input
            if not self.validate_input(input_data):
                return None, None

            prediction = self.model.predict(input_data)
            prediction_prob = self.model.predict_proba(input_data)[:, 1]

            return prediction[0], prediction_prob[0]
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            st.error(f"Prediction error: {str(e)}")
            return None, None


def create_risk_gauge(risk_percentage: float) -> go.Figure:
    """Create an interactive risk gauge using Plotly"""

    # Determine color based on risk level
    if risk_percentage < RISK_THRESHOLDS['low']:
        color = '#4CAF50'  # Green
        risk_level = 'Low'
    elif risk_percentage < RISK_THRESHOLDS['moderate']:
        color = '#FF9800'  # Orange
        risk_level = 'Moderate'
    else:
        color = '#F44336'  # Red
        risk_level = 'High'

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Heart Disease Risk Level: {risk_level}"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"}
    )

    return fig


def create_risk_factors_chart(input_data: Dict[str, Any]) -> go.Figure:
    """Create a chart showing risk factors"""

    # Define risk factors and their weights (simplified)
    risk_factors = {
        'Age': 1 if input_data['age_category'] in ['60-64', '65-69', '70-74', '75-79', '80 or older'] else 0,
        'BMI': 1 if input_data['bmi'] > 30 else 0,
        'Smoking': 1 if input_data['smoking'] == 'Yes' else 0,
        'Diabetes': 1 if input_data['diabetic'] == 'Yes' else 0,
        'Stroke History': 1 if input_data['stroke'] == 'Yes' else 0,
        'Physical Inactivity': 1 if input_data['physical_activity'] == 'No' else 0,
        'Poor Sleep': 1 if input_data['sleep_time'] < 6 or input_data['sleep_time'] > 9 else 0,
        'Alcohol': 1 if input_data['alcohol_drinking'] == 'Yes' else 0,
    }

    factors = list(risk_factors.keys())
    values = list(risk_factors.values())
    colors = ['red' if v == 1 else 'green' for v in values]

    fig = go.Figure(data=[
        go.Bar(x=factors, y=values, marker_color=colors)
    ])

    fig.update_layout(
        title="Your Risk Factors Profile",
        xaxis_title="Risk Factors",
        yaxis_title="Present (1) / Absent (0)",
        height=400,
        showlegend=False
    )

    return fig


def get_personalized_recommendations(input_data: Dict[str, Any], risk_percentage: float) -> list:
    """Generate personalized health recommendations"""
    recommendations = []

    if input_data['bmi'] > 30:
        recommendations.append("🏃‍♂️ Consider weight management through diet and exercise")

    if input_data['smoking'] == 'Yes':
        recommendations.append("🚭 Smoking cessation is the most important step you can take")

    if input_data['physical_activity'] == 'No':
        recommendations.append("💪 Aim for at least 150 minutes of moderate exercise per week")

    if input_data['sleep_time'] < 6 or input_data['sleep_time'] > 9:
        recommendations.append("😴 Maintain 7-9 hours of quality sleep per night")

    if input_data['alcohol_drinking'] == 'Yes':
        recommendations.append("🍷 Limit alcohol consumption to moderate levels")

    if input_data['diabetic'] == 'Yes':
        recommendations.append("🩺 Keep diabetes well-controlled with regular monitoring")

    if risk_percentage > 50:
        recommendations.append("👨‍⚕️ Schedule regular check-ups with your healthcare provider")
        recommendations.append("🩺 Consider cardiovascular screening tests")

    if not recommendations:
        recommendations.append("✅ Keep up your healthy lifestyle!")
        recommendations.append("🔄 Regular health check-ups are still important")

    return recommendations


def main():
    """Main application function"""

    # Initialize the predictor
    predictor = HeartDiseasePredictor()
    model = predictor.load_model()
    predictor.model = model

    # App header
    st.markdown("<h1 class='main-header'>❤️ Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)

    # App description with enhanced styling
    with st.expander("ℹ️ About this app", expanded=False):
        st.markdown("""
        This application uses machine learning to assess your risk of heart disease based on various health factors.

        **How it works:**
        - Enter your health information in the form below
        - Our AI model analyzes your data
        - Get personalized risk assessment and recommendations

        **Features:**
        - Interactive risk visualization
        - Personalized health recommendations
        - Risk factor analysis
        - Educational content
        """)

        st.markdown("""
        <div class='disclaimer'>
        <strong>⚠️ Important Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. 
        Always consult with a healthcare professional for medical advice, diagnosis, or treatment.
        </div>
        """, unsafe_allow_html=True)

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🔍 Risk Assessment", "📊 Analytics", "📚 Learn More"])

    with tab1:
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])

        with col1:
            # Form sections with better organization
            st.markdown("<h2 class='sub-header'>📋 Health Assessment Form</h2>", unsafe_allow_html=True)

            # Demographic section
            with st.container():
                st.markdown("<div class='form-section'>", unsafe_allow_html=True)
                st.markdown("**👤 Personal Information**")
                demo_col1, demo_col2, demo_col3 = st.columns(3)
                with demo_col1:
                    age_category = st.selectbox('Age Category', [
                        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                        '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'],
                                                help="Select your age range")
                with demo_col2:
                    sex = st.selectbox('Sex', ['Male', 'Female'])
                with demo_col3:
                    race = st.selectbox('Race', ['White', 'Black', 'Asian', 'Other'])
                st.markdown("</div>", unsafe_allow_html=True)

            # Physical metrics section
            with st.container():
                st.markdown("<div class='form-section'>", unsafe_allow_html=True)
                st.markdown("**📏 Physical Metrics**")
                phys_col1, phys_col2 = st.columns(2)
                with phys_col1:
                    bmi = st.number_input('BMI', min_value=10.0, max_value=100.0, value=25.0,
                                          help="Body Mass Index (weight in kg / height in m²)")
                    if bmi < 18.5:
                        st.info("BMI suggests underweight")
                    elif bmi > 30:
                        st.warning("BMI suggests obesity - increased heart disease risk")
                with phys_col2:
                    sleep_time = st.slider('Sleep Time (Hours)', min_value=1, max_value=24, value=8,
                                           help="Average hours of sleep per night")
                st.markdown("</div>", unsafe_allow_html=True)

            # Health conditions section
            with st.container():
                st.markdown("<div class='form-section'>", unsafe_allow_html=True)
                st.markdown("**🏥 Health Conditions**")
                health_col1, health_col2 = st.columns(2)

                with health_col1:
                    stroke = st.selectbox('History of Stroke', ['No', 'Yes'])
                    diabetic = st.selectbox('Diabetic', ['No', 'Yes'])
                    asthma = st.selectbox('Asthma', ['No', 'Yes'])

                with health_col2:
                    kidney_disease = st.selectbox('Kidney Disease', ['No', 'Yes'])
                    skin_cancer = st.selectbox('Skin Cancer', ['No', 'Yes'])
                    diff_walking = st.selectbox('Difficulty Walking', ['No', 'Yes'])
                st.markdown("</div>", unsafe_allow_html=True)

            # Lifestyle section
            with st.container():
                st.markdown("<div class='form-section'>", unsafe_allow_html=True)
                st.markdown("**🎯 Lifestyle Factors**")
                life_col1, life_col2 = st.columns(2)

                with life_col1:
                    smoking = st.selectbox('Smoking', ['No', 'Yes'])
                    alcohol_drinking = st.selectbox('Alcohol Drinking', ['No', 'Yes'])

                with life_col2:
                    physical_activity = st.selectbox('Regular Physical Activity', ['No', 'Yes'])
                    gen_health = st.selectbox('General Health', ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
                st.markdown("</div>", unsafe_allow_html=True)

            # Health metrics
            with st.container():
                st.markdown("<div class='form-section'>", unsafe_allow_html=True)
                st.markdown("**📊 Health Metrics**")
                metrics_col1, metrics_col2 = st.columns(2)

                with metrics_col1:
                    physical_health = st.slider('Physical Health Issues (Days/Month)',
                                                min_value=0.0, max_value=30.0, value=0.0,
                                                help="Days in the past 30 days when physical health was not good")

                with metrics_col2:
                    mental_health = st.slider('Mental Health Issues (Days/Month)',
                                              min_value=0.0, max_value=30.0, value=0.0,
                                              help="Days in the past 30 days when mental health was not good")
                st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # Prediction section
            st.markdown("<h2 class='sub-header'>🎯 Risk Assessment</h2>", unsafe_allow_html=True)

            # Create the input data dictionary
            input_dict = {
                'age_category': age_category,
                'sex': sex,
                'bmi': bmi,
                'physical_health': physical_health,
                'mental_health': mental_health,
                'smoking': smoking,
                'alcohol_drinking': alcohol_drinking,
                'stroke': stroke,
                'diff_walking': diff_walking,
                'race': race,
                'diabetic': diabetic,
                'physical_activity': physical_activity,
                'gen_health': gen_health,
                'sleep_time': sleep_time,
                'asthma': asthma,
                'kidney_disease': kidney_disease,
                'skin_cancer': skin_cancer
            }

            # Create input DataFrame
            def create_input_data():
                return pd.DataFrame({
                    'AgeCategory': [age_category],
                    'Sex': [sex],
                    'BMI': [bmi],
                    'PhysicalHealth': [physical_health],
                    'MentalHealth': [mental_health],
                    'Smoking': [smoking],
                    'AlcoholDrinking': [alcohol_drinking],
                    'Stroke': [stroke],
                    'DiffWalking': [diff_walking],
                    'Race': [race],
                    'Diabetic': [diabetic],
                    'PhysicalActivity': [physical_activity],
                    'GenHealth': [gen_health],
                    'SleepTime': [sleep_time],
                    'Asthma': [asthma],
                    'KidneyDisease': [kidney_disease],
                    'SkinCancer': [skin_cancer]
                })

            # Prediction button
            if st.button('🔍 Analyze My Risk', key='predict_button'):
                if predictor.model is None:
                    st.error("Model not available. Please check the model file.")
                else:
                    with st.spinner('Analyzing your data...'):
                        input_data = create_input_data()
                        prediction, prediction_prob = predictor.predict(input_data)

                        if prediction is not None and prediction_prob is not None:
                            risk_percentage = prediction_prob * 100

                            # Determine risk level and styling
                            if risk_percentage < RISK_THRESHOLDS['low']:
                                risk_class = 'low-risk'
                                risk_level = 'Low'
                                risk_color = '#2E7D32'
                            elif risk_percentage < RISK_THRESHOLDS['moderate']:
                                risk_class = 'moderate-risk'
                                risk_level = 'Moderate'
                                risk_color = '#F57C00'
                            else:
                                risk_class = 'high-risk'
                                risk_level = 'High'
                                risk_color = '#D32F2F'

                            # Display results
                            st.markdown(f"""
                            <div class='result-box {risk_class}'>
                                <h3 style='color: {risk_color}; text-align: center;'>{risk_level} Heart Disease Risk</h3>
                                <p style='text-align: center; font-size: 1.2em;'>
                                    Risk Probability: <strong>{risk_percentage:.1f}%</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                            # Interactive gauge
                            fig_gauge = create_risk_gauge(risk_percentage)
                            st.plotly_chart(fig_gauge, use_container_width=True)

                            # Personalized recommendations
                            st.markdown("### 💡 Personalized Recommendations")
                            recommendations = get_personalized_recommendations(input_dict, risk_percentage)
                            for rec in recommendations:
                                st.markdown(f"- {rec}")

                            # Save prediction to session state for analytics tab
                            st.session_state['last_prediction'] = {
                                'risk_percentage': risk_percentage,
                                'input_data': input_dict,
                                'timestamp': datetime.now()
                            }

            # Information box
            st.markdown("""
            <div class='info-box'>
                <strong>🔍 Understanding Risk Factors</strong><br><br>
                <strong>Major Risk Factors:</strong><br>
                • Age (65+ years)<br>
                • Family history<br>
                • Smoking<br>
                • High blood pressure<br>
                • High cholesterol<br>
                • Diabetes<br>
                • Obesity (BMI > 30)<br><br>
                <strong>Lifestyle Factors:</strong><br>
                • Physical inactivity<br>
                • Poor diet<br>
                • Excessive alcohol<br>
                • Stress<br>
                • Poor sleep quality
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>📊 Risk Analytics</h2>", unsafe_allow_html=True)

        if 'last_prediction' in st.session_state:
            pred_data = st.session_state['last_prediction']

            col1, col2 = st.columns(2)

            with col1:
                # Risk factors chart
                fig_factors = create_risk_factors_chart(pred_data['input_data'])
                st.plotly_chart(fig_factors, use_container_width=True)

            with col2:
                # Risk metrics
                st.markdown("### 📈 Risk Metrics")
                risk_pct = pred_data['risk_percentage']

                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Risk Level", f"{risk_pct:.1f}%",
                              delta=f"{risk_pct - 50:.1f}% vs avg")
                with metrics_col2:
                    population_avg = 50  # Assumed population average
                    relative_risk = risk_pct / population_avg
                    st.metric("Relative Risk", f"{relative_risk:.1f}x",
                              delta=f"{'Higher' if relative_risk > 1 else 'Lower'} than average")
        else:
            st.info("Complete a risk assessment in the first tab to see analytics here.")

    with tab3:
        st.markdown("<h2 class='sub-header'>📚 Educational Resources</h2>", unsafe_allow_html=True)

        edu_col1, edu_col2 = st.columns(2)

        with edu_col1:
            st.markdown("""
            ### 🫀 About Heart Disease

            Heart disease refers to several types of heart conditions, including:
            - **Coronary artery disease** - Most common type
            - **Heart failure** - Heart can't pump blood effectively
            - **Arrhythmias** - Irregular heartbeats
            - **Heart valve problems**

            ### 🎯 Prevention Strategies

            **Primary Prevention:**
            - Healthy diet (Mediterranean, DASH)
            - Regular physical activity
            - Maintain healthy weight
            - Don't smoke
            - Limit alcohol
            - Manage stress
            - Get adequate sleep

            **Secondary Prevention:**
            - Take prescribed medications
            - Monitor blood pressure
            - Control cholesterol
            - Manage diabetes
            - Regular medical check-ups
            """)

        with edu_col2:
            st.markdown("""
            ### 🚨 Warning Signs

            **Seek immediate medical attention if you experience:**
            - Chest pain or discomfort
            - Shortness of breath
            - Pain in arms, back, neck, jaw
            - Nausea or vomiting
            - Dizziness or fainting
            - Cold sweats

            ### 📊 Statistics

            - Heart disease is the leading cause of death globally
            - 1 in 4 deaths are caused by heart disease
            - Many heart diseases are preventable
            - Early detection saves lives

            ### 🩺 Screening Tests

            **Regular screening may include:**
            - Blood pressure measurement
            - Cholesterol testing
            - Electrocardiogram (ECG)
            - Stress testing
            - Echocardiogram
            - CT angiography
            """)

        # Interactive educational content
        st.markdown("### 🧮 BMI Calculator")
        calc_col1, calc_col2, calc_col3 = st.columns(3)

        with calc_col1:
            height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        with calc_col2:
            weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
        with calc_col3:
            if height_cm and weight_kg:
                bmi_calc = weight_kg / ((height_cm / 100) ** 2)
                st.metric("Your BMI", f"{bmi_calc:.1f}")

                if bmi_calc < 18.5:
                    st.info("Underweight")
                elif bmi_calc < 25:
                    st.success("Normal weight")
                elif bmi_calc < 30:
                    st.warning("Overweight")
                else:
                    st.error("Obese")

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        <small>
            Heart Disease Risk Predictor | Last updated: {datetime.now().strftime('%Y-%m-%d')}<br>
            This tool is for educational purposes only. Always consult with healthcare professionals for medical advice.
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()