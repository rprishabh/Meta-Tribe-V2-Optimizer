
import streamlit as st
import numpy as np
import pandas as pd
import time

# --- 1. CORE LOGIC FUNCTIONS ---

def get_network_aggregation(predictions):
    n_vertices = predictions.shape[1]
    # Mapping networks to simple terms for marketers
    networks = {
        "Visual (Eyes-on-Screen)": (0, 3000),
        "Auditory (Sound/Speech)": (3000, 6000),
        "Attention (Focus Level)": (9000, 12000),
        "Executive (Information Load)": (12000, 15000),
        "Limbic (Emotional Impact)": (18000, n_vertices)
    }
    return pd.DataFrame({name: predictions[:, s:e].mean(axis=1) for name, (s, e) in networks.items()})

def generate_marketing_report(df_networks, target_strategy):
    means = df_networks.mean()
    
    # 1. Social Media Suitability
    suitability = []
    if means["Limbic (Emotional Impact)"] > 0.6: suitability.append("Instagram/TikTok (High Emotion)")
    if means["Executive (Information Load)"] > 0.6: suitability.append("LinkedIn/YouTube (High Information)")
    if not suitability: suitability.append("General Social Media")

    # 2. Actionable Suggestions
    suggestions = []
    # Hook Analysis (First 3 seconds simulated)
    hook_attention = df_networks["Attention (Focus Level)"].iloc[:3].mean()
    if hook_attention < 0.5:
        suggestions.append("⚠️ **Weak Hook**: The first 3 seconds aren't grabbing attention. Add a faster visual transition or a bold headline.")
    else:
        suggestions.append("✅ **Strong Hook**: Initial engagement is high.")

    # Length Analysis
    if df_networks["Attention (Focus Level)"].iloc[-5:].mean() < 0.4:
        suggestions.append("📉 **Attention Drop**: Users are losing interest at the end. Consider shortening the video by 20%.")

    return {
        "platforms": ", ".join(suitability),
        "suggestions": suggestions,
        "overall_score": int(means.mean() * 100)
    }

def simulate_brain_responses(n_timesteps=15, bias_type='general'):
    base = np.random.rand(n_timesteps, 20484)
    if bias_type == 'video': base[:, 0:3000] += 0.3 
    elif bias_type == 'audio' or bias_type == 'text': base[:, 3000:6000] += 0.3 
    return np.clip(base, 0, 1)

# --- 2. STREAMLIT UI ---

st.set_page_config(page_title="TRIBE v2 Content Optimizer", page_icon="🧠", layout="wide")

st.title("🧠 TRIBE v2: Marketing Content Auditor")
st.markdown("Translate complex brain activity into high-performing content decisions.")

with st.sidebar:
    st.header("📥 Upload Creative")
    content_type = st.radio("Modality", ["Video", "Audio", "Text"])
    
    uploaded_file = None
    text_input = ""
    if content_type == "Video": uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov"])
    elif content_type == "Audio": uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])
    else: text_input = st.text_area("Paste Script")

    st.divider()
    st.header("🎯 Campaign Goal")
    strategy = st.selectbox("Strategy", ['entertainment', 'educational'])
    
    is_ready = uploaded_file is not None or (content_type == "Text" and len(text_input) > 10)
    run_eval = st.button("Run Marketing Audit", type="primary", disabled=not is_ready)

if run_eval:
    with st.spinner("AI is scanning your content for neural triggers..."):
        time.sleep(2)
        preds = simulate_brain_responses(n_timesteps=20, bias_type=content_type.lower())
        df_nets = get_network_aggregation(preds)
        audit = generate_marketing_report(df_nets, strategy)

        st.success("Audit Complete!")

        # --- Marketing Dashboard ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Content Power Score", f"{audit['overall_score']}/100")
        col2.metric("Best Platform", audit['platforms'])
        col3.metric("Analysis Window", "20 Seconds")

        st.divider()
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("📢 Marketing Recommendations")
            for rec in audit['suggestions']:
                st.markdown(rec)
            
            if audit['overall_score'] > 60:
                st.success("🚀 Result: Good to go for publishing!")
            else:
                st.warning("🛠 Result: Needs edits before publishing.")

        with c2:
            st.subheader("🧠 Why this happens (Neuro-Data)")
            st.write("Our model predicts high activation in the following areas:")
            st.json(df_nets.mean().to_dict())

        st.divider()
        st.subheader("📉 Audience Engagement Timeline")
        st.markdown("*This graph shows how focus and emotion change every second. Look for dips to find where people might stop watching.*")
        st.line_chart(df_nets)

else:
    st.info("👈 Upload your video/image/text to see if your content is ready for social media.")
