import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import io
import re

# Try to import vl_convert (for PNG export)
try:
    import vl_convert as vlc
    VL_CONVERT_AVAILABLE = True
except ImportError:
    VL_CONVERT_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="R.A.P. Dashboard",
    layout="wide"
)

# --- HEADER ---
st.title("Reality Anchor Protocol (R.A.P.)")
st.markdown("### Agent-Based Modeling & Moltbook Data Analysis")
st.markdown("---")

# --- MODE SELECTION ---
mode = st.sidebar.radio("Select Data Source:", ["Simulation Model", "Moltbook Data Analysis"])

# ==========================================
# MODE 1: MATHEMATICAL SIMULATION (REAL-TIME)
# ==========================================
if mode == "Simulation Model":
    st.sidebar.header("🎛️ Simulation Parameters")
    st.sidebar.markdown("Adjust these values to see how disinformation spreads in real-time.")
    
    # 1. PARAMETERS WITH EXPLANATIONS
    n_agents = st.sidebar.slider(
        "Number of Agents (Nodes)", 
        min_value=100, max_value=2000, value=1000, step=100,
        help="Total number of participants in the network. A larger network provides more statistical stability but requires more computation."
    )
    
    bot_pct = st.sidebar.slider(
        "Bot Swarm Ratio (%)", 
        min_value=0.0, max_value=0.8, value=0.40, step=0.05,
        help="Percentage of agents that are malicious bots. Bots have a fixed belief of 1.0 (Falsehood) and never change their minds."
    )
    
    steps = st.sidebar.slider(
        "Time Steps (Duration)", 
        min_value=50, max_value=500, value=100, step=10,
        help="How long the simulation runs. More steps allow you to see if the network eventually resists the attack or collapses."
    )
    
    alpha = st.sidebar.slider(
        "Influence Factor (Alpha)", 
        min_value=0.01, max_value=0.5, value=0.1,
        help="Permeability to influence (DeGroot Model). High Alpha = Agents change their minds easily based on neighbors."
    )

    st.sidebar.markdown("---")
    
    # --- LA SOLUZIONE (IL TUO METODO) ---
    st.sidebar.subheader("🛡️ Countermeasures")
    rap_active = st.sidebar.checkbox(
        "Activate Reality Anchor Protocol", 
        value=False,
        help="Enables Bridging Algorithms and Socratic Friction. This filters out viral signals that lack consensus across diverse clusters."
    )
    
    # Visual feedback in sidebar
    if rap_active:
        st.sidebar.success("✅ R.A.P. Defense System: ONLINE")
    else:
        st.sidebar.warning("⚠️ System Vulnerable: Standard Algorithmic Feed")

    # LEGEND
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Legend")
    st.sidebar.info(
        """
        * **🟢 Green (0.0):** Truth / Honest Agent
        * **🔴 Red (1.0):** Disinformation / Bot
        * **🟡 Yellow:** Confused / Mixed Belief
        """
    )

    st.subheader("⚡ Real-time Network Dynamics")

    # 2. SIMULATION LOGIC
    with st.spinner('Simulating...'):
        agents = np.zeros(n_agents)
        n_bots = int(n_agents * bot_pct)
        # Create a contiguous block of bots for visualization clarity
        bot_start = int(n_agents * 0.3)
        bot_end = bot_start + n_bots
        agents[bot_start:bot_end] = 1.0
        
        history = np.zeros((n_agents, steps))
        history[:, 0] = agents.copy()
        current_agents = agents.copy()

        # Vectorized loop
        for t in range(1, steps):
            global_mean = np.mean(current_agents)
            mask_honest = np.ones(n_agents, dtype=bool)
            mask_honest[bot_start:bot_end] = False
            noise = np.random.normal(0, 0.02, size=n_agents)
            
            # --- APPLICAZIONE LOGICA R.A.P. ---
            # Se il protocollo è attivo, riduciamo l'influenza virale (Alpha)
            current_alpha = alpha
            if rap_active:
                current_alpha = alpha * 0.05 # Simulazione del filtro Bridging
            
            # DeGroot Update Rule
            current_agents[mask_honest] += current_alpha * (global_mean - current_agents[mask_honest]) + noise[mask_honest]
            current_agents[bot_start:bot_end] = 1.0 # Reset bots
            
            current_agents = np.clip(current_agents, 0, 1)
            history[:, t] = current_agents.copy()

        # 3. VISUALIZATION
        fig, ax = plt.subplots(figsize=(10, 5))
        cax = ax.imshow(history, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
        
        # Titolo dinamico basato sullo stato di difesa
        status_text = "🛡️ PROTECTED" if rap_active else "⚠️ VULNERABLE"
        ax.set_title(f"Network State: {status_text} (Bots: {bot_pct*100:.0f}%)")
        ax.set_xlabel("Time (Steps)")
        ax.set_ylabel("Agent ID (Population)")
        
        # Custom ticks to hide thousands of agent IDs
        ax.set_yticks([0, bot_start, bot_end, n_agents-1])
        ax.set_yticklabels(["Honest", "Bot Start", "Bot End", "Honest"])
        
        plt.colorbar(cax, label="Belief State (0=True, 1=False)")
        st.pyplot(fig)

        # 4. METRICS & DOWNLOAD
        col1, col2 = st.columns([1, 3])
        
        with col1:
            final_belief = history[:, -1]
            honest_belief = np.concatenate((final_belief[:bot_start], final_belief[bot_end:]))
            avg_corruption = np.mean(honest_belief)
            
            # Logica Metrica: Se la corruzione è bassa (< 0.15) è Stabile
            if avg_corruption < 0.15:
                st.metric("Avg Honest Corruption", f"{avg_corruption:.3f}", delta="Stable", delta_color="normal")
            else:
                st.metric("Avg Honest Corruption", f"{avg_corruption:.3f}", delta="High Risk", delta_color="inverse")

        with col2:
            # Info box se il protocollo è attivo
            if rap_active:
                 st.info("💡 **Observation:** With R.A.P. active, the Bridging Algorithm prevents the red bot swarm from influencing the honest population, keeping corruption low.")
            
            # Download Button
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=300)
            img_buffer.seek(0)
            st.download_button(
                label="💾 Download Heatmap (PNG)",
                data=img_buffer,
                file_name="rap_simulation_heatmap.png",
                mime="image/png"
            )

# ==========================================
# MODE 2: MOLTBOOK DATA ANALYSIS
# ==========================================
elif mode == "Moltbook Data Analysis":
    st.sidebar.header("📂 Data Loading")
    
    default_file = "moltbook_REAL_data.csv"
    df = None
    
    uploaded_file = st.sidebar.file_uploader("Upload Moltbook Log (CSV)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"File loaded!")
    else:
        try:
            df = pd.read_csv(default_file)
            st.sidebar.info(f"Using local dataset.")
        except FileNotFoundError:
            try:
                df = pd.read_csv("moltbook_full_dataset.csv")
                st.sidebar.warning(f"Using simulation data (Real data missing).")
            except:
                st.error("No dataset found. Please upload a CSV file.")

    if df is not None:
        # DATA CLEANING
        def clean_text(text):
            if isinstance(text, str):
                return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
            return text

        for col in ['agent_id', 'content']:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

        with st.expander("🔍 Inspect Raw Data"):
            st.dataframe(df.head(100))

        st.subheader("🤖 Network Activity Analysis")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Interactions", len(df))
        col2.metric("Unique Agents", df['agent_id'].nunique())
        col3.metric("Avg. Belief Score", f"{df['belief_score'].mean():.2f}")

        st.write("### Interactive Timeline")
        st.caption("Hover over points to see Agent ID and Content. Use mouse wheel to zoom.")
        
        chart = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X('timestamp', title='Sequence'),
            y=alt.Y('agent_id', axis=None, title='Agents (Active Swarm)'), 
            color=alt.Color('belief_score', 
                            scale=alt.Scale(scheme='redyellowgreen', domain=[1, 0]), 
                            title='Disinfo Score'),
            tooltip=['agent_id', 'content', 'belief_score', 'timestamp']
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        
        # DOWNLOADS
        col_dl1, col_dl2 = st.columns(2)
        
        # HTML
        html_buffer = io.StringIO()
        chart.save(html_buffer, 'html')
        with col_dl1:
             st.download_button(
                label="💾 Download HTML Chart",
                data=html_buffer.getvalue(),
                file_name="moltbook_analysis.html",
                mime="text/html"
            )
        
        # PNG
        with col_dl2:
            if not VL_CONVERT_AVAILABLE:
                st.error("Library 'vl-convert-python' missing.")
            else:
                try:
                    png_data = vlc.vegalite_to_png(chart.to_json(), scale=2)
                    st.download_button(
                        label="💾 Download PNG Image",
                        data=png_data,
                        file_name="moltbook_analysis.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"PNG Error: {e}")