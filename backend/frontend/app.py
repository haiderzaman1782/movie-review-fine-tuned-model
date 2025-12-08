import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import time

# ============== PAGE CONFIGURATION ==============

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CUSTOM CSS ==============

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.8);
    }
    
    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .positive-card {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10b981;
    }
    
    .negative-card {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid #ef4444;
    }
    .neutral-card {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        border-left: 5px solid #6b7280;
    }
    
    .sentiment-label {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .positive-label {
        color: #059669;
    }
    
    .negative-label {
        color: #dc2626;
    }
     .neutral-label {
        color: #374151;
    }
    
    .confidence-text {
        font-size: 1rem;
        color: #4b5563;
    }
    
    .review-text {
        font-style: italic;
        color: #374151;
        margin-top: 1rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 8px;
    }
    
    /* Status indicator */
    .status-online {
        background: #d1fae5;
        color: #059669;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-offline {
        background: #fee2e2;
        color: #dc2626;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Summary cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============== API CONFIGURATION ==============

API_URL = "https://haider32-movie-analysis.hf.space"

# ============== HELPER FUNCTIONS ==============

def check_api_health() -> bool:
    """Check if the backend API is running."""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_single(text: str) -> Dict:
    """Get prediction for a single review."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Unknown error")}
    except Exception as e:
        return {"error": str(e)}


def predict_batch(reviews: List[str]) -> Dict:
    """Get predictions for multiple reviews."""
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"reviews": reviews},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Unknown error")}
    except Exception as e:
        return {"error": str(e)}


def display_result_card(result: Dict):
    """Display a single result as a styled card."""
    sentiment = result["sentiment"]
    
    if sentiment == "POSITIVE":
        card_class = "positive-card"
        label_class = "positive-label"
        emoji = "😊"
    elif sentiment == "NEGATIVE":
        card_class = "negative-card"
        label_class = "negative-label"
        emoji = "😞"
    else: # NEUTRAL
        card_class = "neutral-card"
        label_class = "neutral-label"
        emoji = "😐"
    
    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="sentiment-label {label_class}">
            {emoji} {sentiment}
        </div>
        <div class="confidence-text">
            Confidence: <strong>{result["confidence"]}%</strong>
        </div>
        <div class="review-text">
            "{result["text"]}"
        </div>
    </div>
    """, unsafe_allow_html=True)
    
def create_confidence_gauge(confidence: float, sentiment: str) -> go.Figure:
    """Create a gauge chart for confidence score."""
    if sentiment == "POSITIVE":
        color = "#10b981"
    elif sentiment == "NEGATIVE":
        color = "#ef4444"
    else: # NEUTRAL
        color = "#6b7280" # Gray
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'size': 20, 'color': 'white'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255,255,255,0.1)'},
                {'range': [50, 75], 'color': 'rgba(255,255,255,0.2)'},
                {'range': [75, 100], 'color': 'rgba(255,255,255,0.3)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_summary_chart(results: List[Dict]) -> go.Figure:
    """Create a pie chart for sentiment distribution."""
    positive = sum(1 for r in results if r["sentiment"] == "POSITIVE")
    negative = sum(1 for r in results if r["sentiment"] == "NEGATIVE")
    neutral = sum(1 for r in results if r["sentiment"] == "NEUTRAL")
    
    fig = px.pie(
        values=[positive, negative, neutral],
        names=["Positive", "Negative", "Neutral"],
        # Green, Red, Gray
        color_discrete_sequence=["#10b981", "#ef4444", "#6b7280"], 
        hole=0.4
    )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=300,
        margin=dict(l=20, r=20, t=20, b=50)
    )
    
    return fig


def create_confidence_histogram(results: List[Dict]) -> go.Figure:
    """Create a histogram of confidence scores."""
    df = pd.DataFrame(results)
    
    fig = px.histogram(
        df,
        x="confidence",
        color="sentiment",
        nbins=20,
        color_discrete_map={
            "POSITIVE": "#10b981", 
            "NEGATIVE": "#ef4444",
            "NEUTRAL": "#6b7280"
        },
        barmode="overlay"
    )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        xaxis_title="Confidence (%)",
        yaxis_title="Count",
        legend_title="Sentiment",
        height=300,
        margin=dict(l=20, r=20, t=20, b=50)
    )
    
    return fig


# ============== MAIN APP ==============

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🎭 Sentiment Analyzer</div>
        <div class="header-subtitle">Powered by Fine-Tuned BERT + LoRA</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # API Status
        st.markdown("### 🔌 API Status")
        api_online = check_api_health()
        
        if api_online:
            st.markdown('<div class="status-online">🟢 Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-offline">🔴 Offline</div>', unsafe_allow_html=True)
            st.error("Backend API is not running!")
            st.code("uvicorn app.main:app --reload --port 8000", language="bash")
        
        st.markdown("---")
        
        # Mode Selection
        st.markdown("### 📝 Analysis Mode")
        mode = st.radio(
            "Select mode:",
            ["Single Review", "Batch Analysis", "CSV Upload"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Example Reviews
        st.markdown("### 💡 Example Reviews")
        example_reviews = [
            "This movie was absolutely fantastic! Best I've seen all year.",
            "Terrible experience. Would not recommend to anyone.",
            "It was okay, nothing special but not bad either.",
            "The acting was superb and the storyline was captivating!",
            "Complete waste of time and money. Very disappointing."
        ]
        
        if st.button("Load Examples"):
            st.session_state.example_loaded = True
            st.session_state.examples = example_reviews
    
    # Main Content
    if not api_online:
        st.warning("⚠️ Please start the backend API to use the sentiment analyzer.")
        st.info("""
        **How to start the backend:**
        1. Open a new terminal
        2. Navigate to the backend folder
        3. Run: `uvicorn app.main:app --reload --port 8000`
        """)
        return
    
    # ============== SINGLE REVIEW MODE ==============
    if mode == "Single Review":
        st.markdown("## 📝 Single Review Analysis")
        st.markdown("Enter a review below to analyze its sentiment.")
        
        # Text Input
        review_text = st.text_area(
            "Enter your review:",
            height=150,
            placeholder="Type or paste a review here...",
            key="single_review"
        )
        
        # Analyze Button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            analyze_btn = st.button("🔍 Analyze", use_container_width=True)
        
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.session_state.single_review = ""
            st.rerun()
        
        if analyze_btn:
            if not review_text.strip():
                st.error("Please enter a review to analyze.")
            else:
                with st.spinner("🔄 Analyzing sentiment..."):
                    result = predict_single(review_text)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.markdown("---")
                    st.markdown("### 📊 Result")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        display_result_card(result)
                    
                    with col2:
                        fig = create_confidence_gauge(result["confidence"], result["sentiment"])
                        st.plotly_chart(fig, use_container_width=True)
    
     # ============== BATCH ANALYSIS MODE ==============
    elif mode == "Batch Analysis":
        st.markdown("## 📦 Batch Analysis")
        st.markdown("Enter multiple reviews (one per line) to analyze them all at once.")
        
        # Load examples if button was clicked
        default_text = ""
        if hasattr(st.session_state, 'example_loaded') and st.session_state.example_loaded:
            default_text = "\n".join(st.session_state.examples)
            st.session_state.example_loaded = False
        
        # Text Input
        reviews_text = st.text_area(
            "Enter reviews (one per line):",
            height=200,
            placeholder="Great product!\nTerrible service.\nIt was okay.",
            value=default_text,
            key="batch_reviews"
        )
        
        # Analyze Button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            analyze_btn = st.button("🔍 Analyze All", use_container_width=True)
        
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.session_state.batch_reviews = ""
            st.rerun()
        
        if analyze_btn:
            reviews = [r.strip() for r in reviews_text.split("\n") if r.strip()]
            
            if not reviews:
                st.error("Please enter at least one review.")
            elif len(reviews) > 100:
                st.error("Maximum 100 reviews allowed at once.")
            else:
                with st.spinner(f"🔄 Analyzing {len(reviews)} reviews..."):
                    result = predict_batch(reviews)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    results = result["results"]
                    
                    st.markdown("---")
                    
                    # Summary Statistics
                    st.markdown("### 📊 Summary")
                    
                    # ✅ UPDATED: Count Neutral separately
                    positive = sum(1 for r in results if r["sentiment"] == "POSITIVE")
                    negative = sum(1 for r in results if r["sentiment"] == "NEGATIVE")
                    neutral = sum(1 for r in results if r["sentiment"] == "NEUTRAL")
                    
                    avg_confidence = sum(r["confidence"] for r in results) / len(results)
                    
                    # ✅ UPDATED: Added 5th column for Neutral
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Total Reviews", len(results))
                    
                    with col2:
                        st.metric("Positive 😊", positive)

                    with col3:
                        st.metric("Neutral 😐", neutral)
                    
                    with col4:
                        st.metric("Negative 😞", negative)
                    
                    with col5:
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                    
                    # Charts
                    st.markdown("### 📈 Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Sentiment Distribution")
                        fig = create_summary_chart(results)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Confidence Distribution")
                        fig = create_confidence_histogram(results)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Results Table
                    st.markdown("### 📋 Detailed Results")
                    
                    df = pd.DataFrame(results)
                    df.index = df.index + 1
                    df.columns = ["Review", "Sentiment", "Confidence (%)"]
                    
                    # ✅ UPDATED: Style logic for Neutral (Gray)
                    def highlight_sentiment(row):
                        val = row["Sentiment"]
                        if val == "POSITIVE":
                            # Light Green
                            return ["background-color: #d1fae5; color: black"] * len(row)
                        elif val == "NEGATIVE":
                            # Light Red
                            return ["background-color: #fee2e2; color: black"] * len(row)
                        else:
                            # Light Gray (Neutral)
                            return ["background-color: #f3f4f6; color: black"] * len(row)
                    
                    styled_df = df.style.apply(highlight_sentiment, axis=1)
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    
                    # Download Button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )
    
    # ============== CSV UPLOAD MODE ==============
    elif mode == "CSV Upload":
        st.markdown("## 📂 CSV Upload Analysis")
        st.markdown("Upload a CSV file with a column containing reviews.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file with a text column containing reviews"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown("### 📄 Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Column Selection
                text_column = st.selectbox(
                    "Select the column containing reviews:",
                    options=df.columns.tolist()
                )
                
                # Limit rows
                max_rows = st.slider(
                    "Number of rows to analyze:",
                    min_value=1,
                    max_value=min(100, len(df)),
                    value=min(10, len(df))
                )
                
                if st.button("🔍 Analyze CSV", use_container_width=True):
                    reviews = df[text_column].head(max_rows).tolist()
                    reviews = [str(r).strip() for r in reviews if pd.notna(r) and str(r).strip()]
                    
                    if not reviews:
                        st.error("No valid reviews found in the selected column.")
                    else:
                        with st.spinner(f"🔄 Analyzing {len(reviews)} reviews..."):
                            result = predict_batch(reviews)
                        
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            results = result["results"]
                            
                            st.markdown("---")
                            
                            # Summary
                            st.markdown("### 📊 Summary")
                            
                            positive = sum(1 for r in results if r["sentiment"] == "POSITIVE")
                            negative = len(results) - positive
                            avg_confidence = sum(r["confidence"] for r in results) / len(results)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total", len(results))
                            with col2:
                                st.metric("Positive", positive)
                            with col3:
                                st.metric("Negative", negative)
                            with col4:
                                st.metric("Avg Conf.", f"{avg_confidence:.1f}%")
                            
                            # Charts
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = create_summary_chart(results)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                fig = create_confidence_histogram(results)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Results
                            results_df = pd.DataFrame(results)
                            results_df.columns = ["Review", "Sentiment", "Confidence"]
                            
                            st.markdown("### 📋 Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6b7280; padding: 1rem;'>"
        "Built with ❤️ using Streamlit + FastAPI + BERT LoRA"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()