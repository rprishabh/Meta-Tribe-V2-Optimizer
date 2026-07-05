
import streamlit as st
import numpy as np
import pandas as pd

# --- 1. CORE LOGIC FUNCTIONS ---

def get_network_aggregation(predictions):
    n_vertices = predictions.shape[1]
    networks = {
        "Visual": (0, 3000), "Auditory/Language": (3000, 6000),
        "Somatomotor": (6000, 9000), "Attention": (9000, 12000),
        "Frontoparietal (Executive)": (12000, 15000),
        "Default Mode (Internal thought)": (15000, 18000), "Limbic (Emotion)": (18000, n_vertices)
    }
    return pd.DataFrame({name: predictions[:, s:e].mean(axis=1) for name, (s, e) in networks.items()})

def generate_physiological_summary(df_networks):
    means = df_networks.mean()
    parts = []
    for net, val in means.items():
        status = "highly stimulated" if val > 0.75 else "moderately stimulated" if val > 0.5 else "showing low engagement"
        parts.append(f"{net} is {status} (score: {val:.2f})")
    return "Physiological Engagement Summary:
- " + "
- ".join(parts)

def assess_content_quality(df_networks, target_type='educational'):
    means = df_networks.mean()
    recs = []
    if target_type == "educational":
        if means["Auditory/Language"] < 0.6: recs.append("Increase narrative depth or verbal explanations.")
        if means["Frontoparietal (Executive)"] < 0.5: recs.append("Introduce complex problems or interactive elements.")
    elif target_type == "entertainment":
        if means["Visual"] < 0.7: recs.append("Enhance visual effects or cinematography.")
        if means["Limbic (Emotion)"] < 0.6: recs.append("Increase dramatic tension or musical cues.")
    return f"### Quality Assessment ({target_type.capitalize()})
" + ("Optimization Suggestions:
- " + "
- ".join(recs) if recs else "Content aligns with neural targets.")

def simulate_brain_responses(n_timesteps=15):
    return np.random.rand(n_timesteps, 20484)

# --- 2. STREAMLIT UI ---

st.set_page_config(page_title="TRIBE v2 Content Optimizer", page_icon="🧠")
st.title("TRIBE v2: Content Evaluation Tool")
st.markdown("Evaluate and optimize your content based on predicted neural responses.")

with st.sidebar:
    st.header("Configuration")
    strategy = st.selectbox("Target Strategy", options=['entertainment', 'educational'])
    duration = st.slider("Stimulus Duration (s)", 5, 60, 15)
    run_eval = st.button("Run Evaluation", type="primary")

if run_eval:
    with st.spinner("Simulating brain responses and analyzing content..."):
        preds = simulate_brain_responses(n_timesteps=duration)
        df_nets = get_network_aggregation(preds)
        summary = generate_physiological_summary(df_nets)
        report = assess_content_quality(df_nets, target_type=strategy)

        st.success("Analysis Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Engagement Summary")
            st.text(summary)
        
        with col2:
            st.subheader("Content Suggestions")
            st.markdown(report)

        st.divider()
        st.subheader("Network Activity Time-course")
        st.line_chart(df_nets)
        st.dataframe(df_nets)
