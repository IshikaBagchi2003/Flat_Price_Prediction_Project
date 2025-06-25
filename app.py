# import streamlit as st
# import numpy as np
# import joblib

# scaller=joblib.load("scaller.pkl")
# model=joblib.load("Model.pkl")
# st.title("Real_Estate_Price_Prediction_Application")

# st.divider()
# bed=st.number_input("enter the no of Bedrooms",value=1,step=1)
# bath=st.number_input("enter the no of Bathroom",value=1,step=1)
# sqft=st.number_input("enter the no of Sqft",value=300,step=50)
# X=[bed,bath,sqft]
# st.divider()
# pred_buttun=st.button("Predict")


# st.divider()

# if pred_buttun:
#     st.balloons()

#     X1=np.array(X)
#     X_array=scaller.fit_transform([X1])
#     y_output=model.predict(X_array)[0]


#     st.write(f"The prediction is { y_output:.2f}")

# else:
#     "Please use the button for prediction"





import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .prediction-result {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("scaller.pkl")
        model = joblib.load("Model.pkl")
        return scaler, model
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'scaller.pkl' and 'Model.pkl' are in the same directory.")
        st.stop()

scaler, model = load_models()

# Header
st.markdown('<h1 class="main-header">🏠 Real Estate Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get instant property value estimates using advanced machine learning</p>', unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.markdown("## 🏡 Property Details")

with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    # Property inputs with better styling
    st.markdown("### 🛏️ Bedrooms")
    bed = st.slider("Number of bedrooms", min_value=1, max_value=10, value=3, step=1)
    
    st.markdown("### 🛁 Bathrooms")
    bath = st.slider("Number of bathrooms", min_value=1, max_value=8, value=2, step=1)
    
    st.markdown("### 📐 Square Footage")
    sqft = st.number_input("Square feet", min_value=300, max_value=10000, value=1500, step=50)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional info
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.info(f"**Price per sq ft estimate:** ${(sqft * 150 if sqft > 0 else 0):,.0f} / {sqft:,} sq ft")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Input summary
    st.markdown("## 📋 Property Summary")
    
    # Create three columns for metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🛏️ Bedrooms", bed)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🛁 Bathrooms", bath)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📐 Square Feet", f"{sqft:,}")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("## 🎯 Make Prediction")
    
    # Prediction button
    predict_button = st.button("🔮 Predict Property Value", use_container_width=True)

# Prediction logic
if predict_button:
    with st.spinner("🔄 Analyzing property data..."):
        # Add a small delay for better UX
        import time
        time.sleep(1)
        
        # Make prediction
        X = [bed, bath, sqft]
        X1 = np.array(X)
        X_array = scaler.transform([X1])  # Use transform instead of fit_transform
        y_output = model.predict(X_array)[0]
        
        # Display results
        st.balloons()
        
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-result">💰 Estimated Price: ${y_output:,.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            price_per_sqft = y_output / sqft if sqft > 0 else 0
            st.metric("💲 Price per Sq Ft", f"${price_per_sqft:.2f}")
        
        with col2:
            # Estimate monthly mortgage (rough calculation)
            monthly_payment = (y_output * 0.8 * 0.05) / 12  # Assuming 20% down, 5% interest
            st.metric("🏦 Est. Monthly Payment", f"${monthly_payment:,.0f}")
        
        with col3:
            # Property category based on price
            if y_output < 200000:
                category = "💡 Affordable"
            elif y_output < 500000:
                category = "🏠 Mid-Range"
            elif y_output < 1000000:
                category = "🏛️ Premium"
            else:
                category = "💎 Luxury"
            st.metric("🏷️ Property Category", category)
        
        # Price visualization
        st.markdown("## 📈 Price Breakdown")
        
        # Create a simple gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = y_output,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Property Value ($)"},
            gauge = {
                'axis': {'range': [None, max(1000000, y_output * 1.2)]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 200000], 'color': "lightgray"},
                    {'range': [200000, 500000], 'color': "gray"},
                    {'range': [500000, 1000000], 'color': "lightblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': y_output
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Market insights
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 💡 Market Insights")
        st.markdown(f"""
        - **Property Type**: {category.split(' ')[1]} range property
        - **Investment Potential**: {'High' if price_per_sqft > 200 else 'Moderate' if price_per_sqft > 100 else 'Good'}
        - **Market Position**: {'Above average' if y_output > 400000 else 'Average' if y_output > 200000 else 'Below average'} for the area
        """)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Default state
    st.markdown("---")
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 🚀 Ready to get started?")
    st.markdown("Adjust the property details in the sidebar and click **Predict Property Value** to get an instant estimate!")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #666; font-size: 0.9rem;'>Last updated: {datetime.now().strftime('%B %d, %Y')} | Powered by Machine Learning 🤖</p>", unsafe_allow_html=True)



