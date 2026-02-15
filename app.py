import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import io
import re
import os
import json
import time
import itertools
import tempfile
import random
from math import pi
from datetime import datetime
from google import genai
from PIL import Image
import pypdf
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from fpdf import FPDF
from wordcloud import WordCloud
import streamlit.components.v1 as components # Added for Auto-Scroll

# Try to import vl_convert (for Chart to Image conversion)
try:
    import vl_convert as vlc
    VL_CONVERT_AVAILABLE = True
except ImportError:
    VL_CONVERT_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RAP Dashboard",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
if 'api_calls' not in st.session_state: st.session_state['api_calls'] = 0
if 'oracle_history' not in st.session_state: st.session_state['oracle_history'] = []
if 'scroll_to_top' not in st.session_state: st.session_state['scroll_to_top'] = False # New state for scrolling

if 'data_store' not in st.session_state:
    st.session_state['data_store'] = {
        'CSV File Upload': {'df': None, 'analyzed': None, 'summary': None},
        'YouTube Link': {'df': None, 'analyzed': None, 'summary': None},
        'Raw Text Paste': {'df': None, 'analyzed': None, 'summary': None},
        'Arena': {'df_a': None, 'df_b': None, 'analyzed_a': None, 'analyzed_b': None}
    }

def increment_counter():
    st.session_state['api_calls'] += 1

# --- HELPER: CALLBACK FOR DATA EDITOR ---
def update_editor_state(method_key):
    """Callback to sync data_editor changes to session_state safely."""
    editor_key = f"editor_{method_key}"
    if editor_key in st.session_state:
        new_data = st.session_state[editor_key]
        if isinstance(new_data, pd.DataFrame):
            st.session_state['data_store'][method_key]['df'] = new_data

# --- HELPER: ROBUST JSON PARSER & REPAIR ---
def extract_json(text):
    try:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```', '', text)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None
    except:
        return None

def sanitize_response(data):
    """Ensures keys exist, handles aliases, and translates defaults."""
    if not data: data = {}
    
    aliases = {
        'explanation': ['analysis', 'reasoning', 'comment', 'description'],
        'correction': ['fix', 'suggestion'],
        'counter_reply': ['reply', 'debunk'],
        'rewritten_text': ['rewrite', 'sanitized_text']
    }
    
    for target, source_list in aliases.items():
        if target not in data or not data[target]:
            for source in source_list:
                if source in data and data[source]:
                    data[target] = data[source]
                    break

    defaults = {
        "has_fallacy": False,
        "fallacy_type": "None",
        "explanation": "Analisi non disponibile.", 
        "correction": "",
        "main_topic": "General",
        "target": "None",
        "primary_emotion": "Neutral",
        "archetype": "Observer",
        "relevance": "Relevant",
        "counter_reply": "",
        "sentiment": "Neutral",
        "aggression": 0,
        "rewritten_text": "Nessuna riscrittura necessaria.",
        "facts": []
    }
    
    for key, default_val in defaults.items():
        if key not in data or data[key] is None:
            data[key] = default_val
        if isinstance(data[key], float) and np.isnan(data[key]):
            data[key] = default_val
        if str(data[key]).lower() == 'nan':
            data[key] = default_val

    if data['explanation'] == "Analisi non disponibile." or data['explanation'] == "No analysis provided.":
        if data.get('has_fallacy'):
            data['explanation'] = f"Rilevata fallacia di tipo: {data.get('fallacy_type')}."
        else:
            data['explanation'] = "Nessuna criticità rilevata, opinione legittima."

    return data

# --- HELPER: VOTE PARSER ---
def parse_votes(vote_str):
    if not vote_str: return 0
    s = str(vote_str).lower().strip()
    if 'k' in s:
        return int(float(re.sub(r'[^\d.]', '', s)) * 1000)
    try:
        return int(re.sub(r'[^\d]', '', s))
    except:
        return 0

# --- HELPER: BOT HUNTER ---
def detect_bot_activity(df):
    if df is None or df.empty: return df
    
    df['is_bot'] = False
    df['bot_reason'] = ""

    dupes = df.duplicated(subset=['content'], keep=False)
    df.loc[dupes, 'is_bot'] = True
    df.loc[dupes, 'bot_reason'] = "Duplicate Content (Coordination)"

    if 'agent_id' in df.columns:
        agent_counts = df['agent_id'].value_counts()
        spammers = agent_counts[agent_counts > 3].index
        df.loc[df['agent_id'].isin(spammers), 'is_bot'] = True
        mask = df['agent_id'].isin(spammers)
        df.loc[mask, 'bot_reason'] = df.loc[mask, 'bot_reason'].apply(lambda x: x + " High Frequency" if x else "High Frequency Spammer")

    return df

# --- HELPER: STRATEGIC SUMMARY ---
def generate_strategic_summary(df, api_key, context="", persona="Intelligence Analyst"):
    if df is None or df.empty: return None
    
    flagged = df[df['has_fallacy']==True]
    bots = df[df['is_bot']==True]
    
    fallacy_counts = flagged['fallacy_type'].value_counts().to_string() if not flagged.empty else "None"
    top_targets = df['target'].value_counts().head(3).to_string() if 'target' in df.columns else "None"
    top_topics = df['main_topic'].value_counts().head(3).to_string() if 'main_topic' in df.columns else "None"
    top_emotion = df['primary_emotion'].mode()[0] if 'primary_emotion' in df.columns and not df['primary_emotion'].empty else "Unknown"
    
    trend_msg = "Stable"
    if 'aggression' in df.columns and len(df) > 10:
        first_half = df['aggression'].iloc[:len(df)//2].mean()
        second_half = df['aggression'].iloc[len(df)//2:].mean()
        if second_half > first_half * 1.2: trend_msg = "ESCALATING (Crisis Risk)"
        elif second_half < first_half * 0.8: trend_msg = "De-escalating"

    bot_count = len(bots)
    avg_agg = df['aggression'].mean() if 'aggression' in df.columns else 0
    
    prompt = f"""
    You are a {persona}. 
    CONTEXT TOPIC: "{context}"
    
    Review data:
    - Avg Aggression: {avg_agg:.1f}/10
    - Trend: {trend_msg}
    - Dominant Emotion: {top_emotion}
    - Bot Activity: {bot_count} items.
    - Fallacies: {fallacy_counts}
    - Topics: {top_topics}
    
    TASK: Write a "Executive Briefing" (max 150 words) from the perspective of a {persona}.
    
    CRITICAL LANGUAGE RULE: 
    - Write in the SAME LANGUAGE as the "CONTEXT TOPIC".
    - If Italian -> Output ITALIAN.
    
    STRUCTURE:
    1. **Overview**: Summary of topic.
    2. **Insight**: Analysis based on your persona ({persona}).
    3. **Forecast**: Trend assessment.
    """
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"Could not generate summary: {str(e)}"

# --- HELPER: THE ORACLE ---
def ask_the_oracle(df, question, api_key, context):
    if df is None: return "No data to analyze."
    
    subset = df[['agent_id', 'content', 'aggression', 'target', 'main_topic', 'archetype']].head(40).to_string()
    
    prompt = f"""
    You are "The Oracle", an advanced intelligence system analyzing a dataset of social media comments.
    CONTEXT OF DISCUSSION: "{context}"
    
    DATASET SAMPLE (Top 40 items):
    {subset}
    
    USER QUESTION: "{question}"
    
    INSTRUCTIONS:
    1. Answer based ONLY on the provided data.
    2. Be concise, professional, and strategic.
    3. Cite specific users or examples if relevant.
    4. RESPOND IN THE SAME LANGUAGE AS THE USER QUESTION.
    """
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"Oracle Error: {str(e)}"

# --- HELPER: EXCEL GENERATOR ---
def generate_excel_report(df, summary_text):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_data = [{'Metric': 'Analysis Date', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M')}]
        if summary_text:
            summary_data.append({'Metric': 'Executive Briefing', 'Value': summary_text})
        
        total = len(df)
        flagged = len(df[df['has_fallacy'] == True])
        
        summary_data.extend([
            {'Metric': 'Total Items', 'Value': total},
            {'Metric': 'Flagged Issues', 'Value': flagged},
            {'Metric': 'Avg Aggression', 'Value': df['aggression'].mean() if 'aggression' in df.columns else 0}
        ])
        
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive Briefing', index=False)
        df.to_excel(writer, sheet_name='Full Analysis', index=False)
    return output.getvalue()

# --- HELPER: PDF GENERATOR ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'RAP: Strategic Intelligence Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(df, chart_obj=None, timeline_obj=None, summary_text=None):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Executive Summary", 0, 1)
    if summary_text:
        pdf.set_font("Helvetica", 'I', 11)
        clean_summ = summary_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 6, clean_summ)
        pdf.ln(5)
    
    if chart_obj and VL_CONVERT_AVAILABLE:
        try:
            png_data = vlc.vegalite_to_png(chart_obj.to_json(), scale=2)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(png_data)
                tmp_path = tmp.name
            pdf.image(tmp_path, w=180)
            os.unlink(tmp_path)
        except: pass
    
    return bytes(pdf.output())

# --- HELPER: PDF EXTRACTOR ---
@st.cache_data
def extract_text_from_pdf(file_obj):
    try:
        pdf_reader = pypdf.PdfReader(file_obj)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content: text += content + "\n"
        return text
    except: return None

# --- HELPER: YOUTUBE SCRAPER ---
@st.cache_data(show_spinner=False)
def scrape_youtube_comments(url, limit=50):
    downloader = YoutubeCommentDownloader()
    comments = []
    try:
        generator = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
        for comment in itertools.islice(generator, limit):
            comments.append({
                'agent_id': comment.get('author', 'Anonymous'),
                'timestamp': comment.get('time', 'Unknown'),
                'content': comment.get('text', ''),
                'likes': parse_votes(comment.get('votes', '0'))
            })
        if not comments: return None
        return pd.DataFrame(comments)
    except: return None

# --- HELPER: PASTE PARSER ---
def parse_raw_paste(raw_text):
    lines = raw_text.split('\n')
    cleaned_data = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5: continue
        cleaned_data.append({'agent_id': 'Paste_Source', 'timestamp': 'Unknown', 'content': line, 'likes': 0})
    return pd.DataFrame(cleaned_data)

# --- HELPER: DF NORMALIZER ---
def normalize_dataframe(df):
    target_cols = {
        'content': ['content', 'text', 'body', 'comment'],
        'agent_id': ['agent_id', 'author', 'user'],
        'timestamp': ['timestamp', 'created_at', 'date'],
        'likes': ['likes', 'votes', 'like_count']
    }
    new_df = df.copy()
    new_df.columns = [c.lower().strip() for c in new_df.columns]
    found_cols = {}
    for target, aliases in target_cols.items():
        for alias in aliases:
            if alias in new_df.columns:
                found_cols[target] = alias
                break
    rename_map = {v: k for k, v in found_cols.items()}
    new_df = new_df.rename(columns=rename_map)
    if 'content' not in new_df.columns:
        string_cols = new_df.select_dtypes(include=['object']).columns
        if len(string_cols) > 0:
            best = max(string_cols, key=lambda c: new_df[c].astype(str).str.len().mean())
            new_df = new_df.rename(columns={best: 'content'})
    if 'agent_id' not in new_df.columns: new_df['agent_id'] = 'Unknown_User'
    if 'timestamp' not in new_df.columns: new_df['timestamp'] = 'Unknown_Time'
    if 'likes' not in new_df.columns: new_df['likes'] = 0
    return new_df[['agent_id', 'timestamp', 'content', 'likes']]

# --- HELPER: HYBRID ANALYZER (MODULE 2) ---
@st.cache_data(show_spinner=False)
def analyze_fallacies_cached(text, api_key, context_info="", persona="Logic & Fact Analysis Engine"):
    if not text or len(str(text)) < 3: return None
    for attempt in range(3):
        try:
            client = genai.Client(api_key=api_key)
            is_long = len(text) > 2000
            
            common_instructions = f"""
            You are a {persona}.
            CONTEXT (Global): "{context_info}"
            
            *** CRITICAL INSTRUCTION: LANGUAGE FORCE ***
            1. **DETECT LANGUAGE**: Identify the language of the 'Text'.
            2. **TRANSLATE EVERYTHING**: Translate 'explanation', 'correction', 'counter_reply' AND CATEGORICAL VALUES (Archetype, Emotion) into the DETECTED LANGUAGE.
            3. **FORBIDDEN**: Do NOT output English explanations/labels if the input is Italian/Spanish etc.
            
            TASKS:
            - **explanation**: If Green, PROVIDE A BRIEF ANALYSIS of the content (e.g. "Valid expression of opinion", "Factual statement") - Translated. DO NOT just write "Reasoning valid".
            - **main_topic**: Central theme (2-3 words, Translated).
            - **target**: Target Entity.
            - **primary_emotion**: Select one: [Anger, Fear, Disgust, Sadness, Joy, Surprise, Neutral] -> TRANSLATE VALUE TO INPUT LANGUAGE.
            - **archetype**: Select one: [Instigator, Loyalist, Troll, Rational Skeptic, Observer] -> TRANSLATE VALUE TO INPUT LANGUAGE.
            
            RULES:
            1. **OPINIONS**: Taste/Praise -> 'has_fallacy': false.
            2. **SARCASM**: Sarcasm -> 'has_fallacy': false.
            
            RESPONSE (Strict JSON):
            {{ 
                "has_fallacy": true/false, 
                "fallacy_type": "Translated Name (e.g. Argomento Fantoccio)", 
                "explanation": "MANDATORY: Detailed Analysis in INPUT LANGUAGE.", 
                "correction": "Correction in INPUT LANGUAGE (if needed)",
                "main_topic": "Theme (Translated)",
                "target": "Target Entity",
                "primary_emotion": "Emotion (Translated)",
                "archetype": "Archetype (Translated)",
                "relevance": "Relevance",
                "counter_reply": "Polite reply in INPUT LANGUAGE",
                "sentiment": "Positive/Neutral/Negative",
                "aggression": 0-10
            }}
            """

            if is_long:
                prompt = f"Analyze LONG DOCUMENT.\n{common_instructions}\nText: \"{text[:15000]}...\""
            else:
                prompt = f"Analyze Text.\n{common_instructions}\nText: \"{text}\""
                
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            result = extract_json(response.text)
            if result: return sanitize_response(result)
        except:
            time.sleep(0.1)
            continue
    return sanitize_response(None)

def analyze_fallacies(text, api_key=None, context_info="", persona="Logic & Fact Analysis Engine"):
    if not api_key: return {"has_fallacy": True}
    increment_counter()
    return analyze_fallacies_cached(text, api_key, context_info, persona)

# --- HELPER: COGNITIVE EDITOR (MODULE 3 - VISION & TEXT) ---
@st.cache_data(show_spinner=False)
def cognitive_rewrite(text, api_key, image_data=None):
    if (not text or len(str(text)) < 3) and not image_data: return None
    try:
        client = genai.Client(api_key=api_key)
        
        prompt_text = f"""
        You are a "Cognitive Editor", "Logic Guard", and "Fact Checker".
        
        TASK:
        1. **DIAGNOSE**: Check for Logical Fallacies, Aggression, AND FACTUAL ERRORS (especially in the image if provided).
        2. **REWRITE**: Create a neutral, logical, and factual version (describe the neutral meaning if image).
        3. **EXTRACT FACTS**: List claims to verify.
        
        CRITICAL RULES:
        1. **FACTUAL ERRORS ARE ISSUES**: If the text contains an objectively false statement (e.g. historical/scientific error), you MUST set "has_fallacy": true and "fallacy_type": "Errore Fattuale" (translated).
        2. **LOGIC**: Flag standard fallacies.
        3. **OPINIONS**: Pure opinions/taste are NOT issues.
        4. **LANGUAGE**: Output strictly in the INPUT LANGUAGE.
        
        RESPONSE (Strict JSON):
        {{
            "has_fallacy": true/false,
            "fallacy_type": "Name (Translated) or None",
            "explanation": "Detailed analysis. If Factual Error, explain the truth.",
            "rewritten_text": "The sanitized/corrected text (Input Language).",
            "facts": ["Claim 1", "Claim 2"],
            "aggression": 0-10
        }}
        """
        
        contents = [prompt_text]
        if text: contents.append(f"Input Text: {text[:10000]}")
        if image_data: contents.append(image_data)
        
        response = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
        res = extract_json(response.text)
        return sanitize_response(res)
    except:
        return sanitize_response(None)

# --- CHARTING HELPERS ---
def plot_radar_chart(df):
    if 'primary_emotion' not in df.columns: return None
    top_emotions = df['primary_emotion'].value_counts().head(7)
    if top_emotions.empty: return None
    labels = top_emotions.index.tolist()
    values = top_emotions.values.flatten().tolist()
    values += values[:1]
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_title("Emotional Radar", size=12, pad=20)
    return fig

def plot_heatmap(df):
    if 'target' not in df.columns or 'main_topic' not in df.columns: return None
    valid = df[(df['target'] != "Unknown") & (df['main_topic'] != "Unknown")]
    if valid.empty: return None
    heatmap = alt.Chart(valid).mark_rect().encode(
        x=alt.X('target:N', title='Target'),
        y=alt.Y('main_topic:N', title='Narrative/Topic'),
        color=alt.Color('count()', title='Count', scale=alt.Scale(scheme='orangered')),
        tooltip=['target', 'main_topic', 'count()']
    ).properties(height=300)
    return heatmap

# --- HEADER ---
st.title("RAP: Reality Anchor Protocol")
st.markdown("### Cognitive Security & Logical Analysis Suite")
st.markdown("---")

mode = st.sidebar.radio("Select Module:", ["Wargame Room (Simulation)", "Social Data Analysis (Universal)", "Comparison Test (A/B Testing)", "Cognitive Editor (Logic Guard)"])

# --- DISCLAIMER ---
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ System Capabilities & Limits", expanded=False):
    st.markdown("""
    **What this system CAN do:**
    - Detect logical fallacies.
    - Analyze sentiment, aggression, and archetypes.
    - Rewrite toxic content (Logic Guard).
    - Simulate network effects (Wargame).
    - Analyze Images (Vision Guard).
    
    **Blind Spots:**
    - **Recent Events:** Knowledge cutoff.
    - **Hallucinations:** AI can make errors. 
    
    *Always verify critical claims independently.*
    """)

st.sidebar.markdown("---")
st.sidebar.caption(f"API Calls Session: {st.session_state['api_calls']}")

# ==========================================
# MODULE 1: WARGAME ROOM (SIMULATION)
# ==========================================
if mode == "Wargame Room (Simulation)":
    st.header("Information Warfare Simulator")
    
    c_param1, c_param2 = st.columns(2)
    with c_param1:
        st.subheader("Network Topology")
        topology = st.selectbox("Scenario Type", ["Public Square (High Connectivity)", "Echo Chambers (Clusters)", "Influencer Network (Hubs)"])
        n_agents = st.slider("Population Size", 100, 2000, 1000)
        bot_pct = st.slider("Infection/Bot Ratio", 0.0, 0.5, 0.10)
    with c_param2:
        st.subheader("Countermeasures (Blue Team)")
        defense = st.selectbox("Active Defense Protocol", ["None (Control Group)", "Fact-Check Debunking (Targeted)", "Algorithmic Dampening (Global)", "Hard Ban (Removal)"])
        steps = st.slider("Simulation Duration (Days)", 50, 300, 100)
        
    # REAL-TIME EXECUTION
    agents = np.zeros(n_agents)
    n_infected = int(n_agents * bot_pct)
    if topology == "Echo Chambers (Clusters)":
        start = int(n_agents * 0.4)
        agents[start : start + n_infected] = 1.0
    else:
        indices = np.random.choice(n_agents, n_infected, replace=False)
        agents[indices] = 1.0
        
    history = np.zeros((n_agents, steps))
    history[:, 0] = agents.copy()
    infection_rate = []
    current = agents.copy()
    
    if topology == "Public Square (High Connectivity)":
        influence_strength = 0.05
        noise_level = 0.02
    elif topology == "Echo Chambers (Clusters)":
        influence_strength = 0.15
        noise_level = 0.01
    else:
        influence_strength = 0.03
        noise_level = 0.01
        
    for t in range(1, steps):
        prev = current.copy()
        if topology == "Echo Chambers (Clusters)":
            left = np.roll(prev, 1)
            right = np.roll(prev, -1)
            neighbor_avg = (left + right) / 2
            current = prev + influence_strength * (neighbor_avg - prev)
        elif topology == "Influencer Network (Hubs)":
            hub_val = prev[0]
            current = prev + influence_strength * (hub_val - prev)
            current[0] = prev[0]
        else:
            global_mean = np.mean(prev)
            current = prev + influence_strength * (global_mean - prev)
        
        noise = np.random.normal(0, noise_level, n_agents)
        current += noise
        
        if defense == "Algorithmic Dampening (Global)": current *= 0.95
        elif defense == "Fact-Check Debunking (Targeted)":
            heal_indices = np.random.choice(n_agents, int(n_agents * 0.02))
            current[heal_indices] = 0.0
        elif defense == "Hard Ban (Removal)":
            current[current > 0.8] = 0.0
        
        current = np.clip(current, 0, 1)
        history[:, t] = current.copy()
        infection_rate.append(np.mean(current))
        
    st.markdown("---")
    c_res1, c_res2 = st.columns([3, 1])
    with c_res1:
        st.subheader("Infection Spread (Heatmap)")
        fig, ax = plt.subplots(figsize=(10, 4))
        cax = ax.imshow(history, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Agent ID")
        st.pyplot(fig)
    with c_res2:
        st.subheader("Infection Rate")
        chart_data = pd.DataFrame({'Time': range(len(infection_rate)), 'Infection Level': infection_rate})
        line = alt.Chart(chart_data).mark_line(color='red').encode(x='Time', y='Infection Level').properties(height=300)
        st.altair_chart(line, use_container_width=True)
        final_rate = infection_rate[-1] * 100
        delta = final_rate - (infection_rate[0] * 100)
        st.metric("Final Infection Level", f"{final_rate:.1f}%", f"{delta:.1f}%", delta_color="inverse")

# ==========================================
# MODULE 2: SOCIAL DATA ANALYSIS
# ==========================================
elif mode == "Social Data Analysis (Universal)":
    st.sidebar.header("Settings")
    persona = st.sidebar.selectbox("Analysis Lens (Persona)", ["Strategic Intelligence Analyst", "Mass Psychologist (Emotional)", "Legal Consultant (Defamation/Risk)", "Campaign Manager (Opportunity)"])
    
    input_method = st.sidebar.radio("Input Method:", ["CSV File Upload", "YouTube Link", "Raw Text Paste"], horizontal=True)
    st.markdown("---")
    context_input = st.text_input("Global Context (Optional)", placeholder="E.g., 'Discussion about Flat Earth'")
    st.caption("Provide context to help AI understand sarcasm and specific facts.")
    st.markdown("---")

    current_storage = st.session_state['data_store'][input_method]
    
    if input_method == "CSV File Upload":
        st.info("Desktop: Use 'Instant Data Scraper' extension.")
        # --- RESTORED INSTRUCTIONS BLOCK ---
        with st.expander("How to extract data from Facebook, X (Twitter), Instagram"):
            st.markdown("Social media platforms do not allow direct downloading. You need a **Browser Extension**.")
            st.markdown("#### Recommended Tools:")
            st.markdown("1. **[Instant Data Scraper](https://chromewebstore.google.com/detail/instant-data-scraper/ofaokhiedipichpaobibbnahnkdoiiah)** (Free & Unlimited)\n*Best for lists of comments or posts. Easy to use (Pokeball icon).*")
            st.markdown("2. **[Export Comments](https://exportcomments.com/)** (Freemium)\n*Easiest for specific posts, but has limits on the free plan.*")
            st.markdown("#### Step-by-Step Guide:")
            st.markdown("1. Install one of the extensions above on Chrome/Edge.")
            st.markdown("2. Go to the post you want to analyze.")
            st.markdown("3. **Crucial:** Scroll down to load all comments *before* starting the extension.")
            st.markdown("4. Click the extension icon and download as **CSV** or **XLSX**.")
            st.markdown("5. Upload the file here. RAP will auto-detect the text.")
        
        with st.form("csv_form"):
            uploaded_file = st.file_uploader("Upload CSV", type="csv")
            submitted = st.form_submit_button("Load Data")
            if submitted and uploaded_file:
                try:
                    raw_df = pd.read_csv(uploaded_file)
                    norm_df = normalize_dataframe(raw_df)
                    norm_df = detect_bot_activity(norm_df)
                    st.session_state['data_store'][input_method]['df'] = norm_df
                    st.success(f"Loaded {len(norm_df)} rows.")
                except: st.error("CSV Error.")

    elif input_method == "YouTube Link":
        with st.form("yt_form"):
            yt_url = st.text_input("YouTube URL")
            limit = st.number_input("Max Comments to Scrape", min_value=1, max_value=2000, value=50, step=10)
            submitted = st.form_submit_button("Scrape")
            if submitted:
                with st.spinner("Scraping..."):
                    sdf = scrape_youtube_comments(yt_url, limit)
                    if sdf is not None:
                        sdf = detect_bot_activity(sdf)
                        st.session_state['data_store'][input_method]['df'] = sdf
                        st.session_state['data_store'][input_method]['analyzed'] = None 
                        st.session_state['data_store'][input_method]['summary'] = None
                        st.success("Scraped!")
                    else: st.error("Failed.")

    elif input_method == "Raw Text Paste":
        with st.form("text_form"):
            raw_text = st.text_area("Paste content", height=150)
            submitted = st.form_submit_button("Process")
            if submitted:
                if raw_text:
                    pdf_parsed = parse_raw_paste(raw_text)
                    pdf_parsed = detect_bot_activity(pdf_parsed)
                    st.session_state['data_store'][input_method]['df'] = pdf_parsed
                    st.session_state['data_store'][input_method]['analyzed'] = None 
                    st.session_state['data_store'][input_method]['summary'] = None
                    st.success(f"Extracted {len(pdf_parsed)} items.")

    df = st.session_state['data_store'][input_method]['df']
    
    if df is not None:
        if isinstance(df, pd.DataFrame):
            def clean(t): return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', str(t))
            if 'content' in df.columns: df['content'] = df['content'].apply(clean)

            st.markdown("---")
            c1, c2 = st.columns([2, 1])
            with c1:
                if 'Select' not in df.columns:
                    df.insert(0, "Select", False)
                
                # --- TABLE WITH CALLBACK FIX (CRASH GUARDED) ---
                editor_output = st.data_editor(
                    st.session_state['data_store'][input_method]['df'],
                    column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                    disabled=["content", "agent_id", "timestamp", "is_bot", "bot_reason", "likes"],
                    use_container_width=True,
                    hide_index=True,
                    key=f"editor_{input_method}",
                    on_change=update_editor_state,
                    args=(input_method,),
                    height=300
                )
                current_df = editor_output if editor_output is not None else df

            with c2:
                st.metric("Items", len(df))
                bots_detected = len(df[df['is_bot']==True])
                st.metric("⚠️ Suspicious Bots", bots_detected)
                if len(df) > 0:
                    try:
                        text_combined = " ".join(df['content'].astype(str).tolist())
                        wc = WordCloud(width=400, height=200, background_color='black', colormap='Reds', random_state=42).generate(text_combined)
                        fig_wc, ax_wc = plt.subplots()
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                    except: st.caption("Not enough text for WordCloud")

            st.markdown("---")
            if "GEMINI_API_KEY" in st.secrets: key = st.secrets["GEMINI_API_KEY"]
            else: key = st.text_input("API Key", type="password")

            with st.form("analysis_exec_form"):
                c_form1, c_form2 = st.columns([1, 4])
                
                if isinstance(current_df, pd.DataFrame) and 'Select' in current_df.columns:
                    selected_rows = current_df[current_df.Select]
                    has_selection = not selected_rows.empty
                else:
                    selected_rows = pd.DataFrame()
                    has_selection = False

                with c_form1:
                    if not has_selection:
                        scan_rows = st.number_input("Max Rows to Analyze", min_value=1, max_value=len(df), value=min(20, len(df)), step=1)
                    else:
                        st.info(f"Manual Mode: {len(selected_rows)} selected.")
                        scan_rows = 0 
                    
                    btn_label = f"RUN ANALYSIS ON SELECTED ({len(selected_rows)})" if has_selection else "RUN ANALYSIS"
                    run_submitted = st.form_submit_button(btn_label, disabled=not key, type="primary")

            if run_submitted:
                prog = st.progress(0)
                res = []
                if has_selection: subset = selected_rows.copy()
                else: subset = current_df.head(scan_rows).copy()
                total_to_scan = len(subset)
                for i, (_, row) in enumerate(subset.iterrows()):
                    ans = analyze_fallacies(row['content'], api_key=key, context_info=context_input, persona=persona)
                    res.append(ans)
                    prog_val = min((i+1)/total_to_scan, 1.0)
                    prog.progress(prog_val)
                    time.sleep(0.1) 
                
                final_df = pd.concat([subset.reset_index(drop=True), pd.DataFrame(res)], axis=1)
                st.session_state['data_store'][input_method]['analyzed'] = final_df
                
                with st.spinner("Generating Strategic Briefing..."):
                    summ = generate_strategic_summary(final_df, key, context=context_input, persona=persona)
                    st.session_state['data_store'][input_method]['summary'] = summ
                
                # TRIGGER AUTO-SCROLL
                st.session_state['scroll_to_top'] = True
                st.rerun()

            analyzed_df = st.session_state['data_store'][input_method]['analyzed']
            summary_text = st.session_state['data_store'][input_method]['summary']

            if analyzed_df is not None:
                # AUTO-SCROLL ANCHOR AND SCRIPT (WITH TIMEOUT FIX)
                st.markdown('<div id="briefing_anchor"></div>', unsafe_allow_html=True)
                
                if st.session_state.get('scroll_to_top'):
                    components.html(
                        """
                        <script>
                            setTimeout(function() {
                                var element = window.parent.document.getElementById("briefing_anchor");
                                if (element) {
                                    element.scrollIntoView({behavior: "smooth", block: "start"});
                                }
                            }, 800); 
                        </script>
                        """,
                        height=0
                    )
                    st.session_state['scroll_to_top'] = False

                adf = analyzed_df
                if summary_text:
                    st.markdown("### Strategic Intelligence Briefing")
                    st.info(summary_text)
                
                st.markdown("### Metrics")
                m1, m2, m3 = st.columns(3)
                flagged_count = len(adf[adf['has_fallacy']==True])
                agg_avg = adf['aggression'].mean()
                m1.metric("Issues Detected", flagged_count)
                m2.metric("Avg Aggression", f"{agg_avg:.1f}/10")
                
                st.markdown("---")
                with st.expander("📊 Open Intelligence Visuals (Radar, Heatmap & Targets)", expanded=False):
                    st.subheader("Narrative & Emotional Intelligence")
                    c_vis1, c_vis2 = st.columns(2)
                    with c_vis1:
                        radar_fig = plot_radar_chart(adf)
                        if radar_fig: st.pyplot(radar_fig)
                        else: st.caption("No emotional data.")
                    with c_vis2:
                        heatmap = plot_heatmap(adf)
                        if heatmap: st.altair_chart(heatmap, use_container_width=True)
                        else: st.caption("No heatmap data.")
                    st.markdown("---")
                    c_intel1, c_intel2 = st.columns(2)
                    with c_intel1:
                        st.caption("Main Narratives (Topics)")
                        if 'main_topic' in adf.columns:
                            top_topics = adf['main_topic'].value_counts().reset_index().head(10)
                            top_topics.columns = ['Topic', 'Count']
                            chart_topic = alt.Chart(top_topics).mark_bar().encode(
                                x='Count', y=alt.Y('Topic', sort='-x'), color=alt.Color('Topic', legend=None)
                            ).properties(height=300)
                            st.altair_chart(chart_topic, use_container_width=True)
                    with c_intel2:
                        st.caption("User Archetypes")
                        if 'archetype' in adf.columns:
                            top_arch = adf['archetype'].value_counts().reset_index()
                            top_arch.columns = ['Archetype', 'Count']
                            chart_arch = alt.Chart(top_arch).mark_bar().encode(
                                x='Count', y=alt.Y('Archetype', sort='-x'), color=alt.Color('Archetype', scale=alt.Scale(scheme='dark2'), legend=None)
                            ).properties(height=300)
                            st.altair_chart(chart_arch, use_container_width=True)

                st.markdown("---")
                c_chart1, c_chart2 = st.columns(2)
                with c_chart1:
                    st.caption("Fallacy Distribution")
                    cnt = adf[adf['has_fallacy']==True]['fallacy_type'].value_counts().reset_index()
                    cnt.columns = ['Type', 'Count']
                    if not cnt.empty:
                        chart = alt.Chart(cnt).mark_bar().encode(
                            x='Count', y=alt.Y('Type', sort='-x'), color=alt.Color('Type', scale=alt.Scale(scheme='reds'))
                        ).properties(height=300)
                        st.altair_chart(chart, use_container_width=True)
                with c_chart2:
                    st.caption("Crisis Trend (Aggression)")
                    adf['Sequence'] = adf.index
                    regression = alt.Chart(adf).mark_line(color='red', strokeDash=[5,5]).transform_regression('Sequence', 'aggression').encode(x='Sequence', y='aggression')
                    line = alt.Chart(adf).mark_line(color='orange').encode(
                        x=alt.X('Sequence', title='Comment Order'),
                        y=alt.Y('aggression', title='Aggression Level (0-10)'),
                        tooltip=['content', 'aggression']
                    )
                    st.altair_chart(line + regression, use_container_width=True)

                st.markdown("---")
                st.subheader("Data Explorer")
                c_filter1, c_filter2 = st.columns(2)
                with c_filter1:
                    filter_fallacy = st.multiselect("Filter by Issue Type", adf['fallacy_type'].unique())
                with c_filter2:
                    min_agg = st.slider("Min Aggression", 0, 10, 0)
                
                view = adf.copy()
                if filter_fallacy: view = view[view['fallacy_type'].isin(filter_fallacy)]
                view = view[view['aggression'] >= min_agg]
                
                for _, r in view.iterrows():
                    with st.container(border=True):
                        c1, c2 = st.columns([0.05, 0.95])
                        status = "🟢"
                        if r['has_fallacy']: status = "🔴"
                        if r.get('is_bot'): status = "🤖"
                        c1.write(status)
                        bot_msg = f" | ⚠️ BOT SUSPECT: {r.get('bot_reason')}" if r.get('is_bot') else ""
                        emotion_tag = f" | {r.get('primary_emotion', '')}" if r.get('primary_emotion') else ""
                        likes_tag = f" | 👍 {r.get('likes', 0)}"
                        arch_tag = f" | 🎭 {r.get('archetype', 'User')}"
                        c2.caption(f"**User:** {r.get('agent_id', 'User')} | **Agg:** {r.get('aggression')}/10 {likes_tag}{emotion_tag}{arch_tag}{bot_msg}")
                        st.info(f"\"{r['content']}\"")
                        if r['has_fallacy']:
                            st.error(f"**{r['fallacy_type']}**: {r['explanation']}")
                            if r.get('counter_reply'):
                                with st.expander("Show Counter-Reply (Debunker)"):
                                    st.markdown(f"**Suggested Reply:**\n> *{r['counter_reply']}*")
                        else:
                            explanation = str(r.get('explanation', ''))
                            if explanation in ["None", "nan", ""]: explanation = "Analisi valida, nessuna criticità rilevata."
                            st.success(f"✅ {explanation}")

                st.markdown("---")
                c_down1, c_down2 = st.columns([1, 1])
                with c_down1:
                    excel_data = generate_excel_report(adf, summary_text)
                    st.download_button("Download Full Excel Report", excel_data, "RAP_Intelligence_Report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with c_down2:
                    if st.button("Generate PDF Report"):
                        pdf_bytes = generate_pdf_report(adf, chart, timeline, summary_text)
                        st.download_button("Download PDF", pdf_bytes, "RAP_Executive_Report.pdf", "application/pdf")
                
                st.markdown("---")
                st.subheader("The Oracle (Chat with Data)")
                st.caption("Ask questions about the analyzed data.")
                for message in st.session_state.oracle_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                if prompt := st.chat_input("Ask the Oracle..."):
                    st.session_state.oracle_history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Oracle is thinking..."):
                            response = ask_the_oracle(adf, prompt, key, context_input)
                            st.markdown(response)
                            st.session_state.oracle_history.append({"role": "assistant", "content": response})
        else:
            st.error("Data integrity check failed. Please reload the file.")

# ==========================================
# MODULE 4: COMPARISON TEST (A/B TESTING)
# ==========================================
elif mode == "Comparison Test (A/B Testing)":
    st.header("Comparison Test (A/B Testing)")
    st.caption("Compare two YouTube videos or Datasets side-by-side.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Dataset A")
        url_a = st.text_input("YouTube URL A", key="url_a")
    with col_b:
        st.subheader("Dataset B")
        url_b = st.text_input("YouTube URL B", key="url_b")
        
    limit_ab = st.number_input("Max Comments per Video", 10, 500, 50)
    
    if "GEMINI_API_KEY" in st.secrets: key = st.secrets["GEMINI_API_KEY"]
    else: key = st.text_input("API Key", type="password")
    
    if st.button("Compare A vs B"):
        if url_a and url_b and key:
            with st.spinner("Analyzing Contender A..."):
                df_a = scrape_youtube_comments(url_a, limit_ab)
                if df_a is not None:
                    res_a = []
                    for i, (_, row) in enumerate(df_a.iterrows()):
                        res_a.append(analyze_fallacies(row['content'], api_key=key))
                    df_a = pd.concat([df_a.reset_index(drop=True), pd.DataFrame(res_a)], axis=1)
            
            with st.spinner("Analyzing Contender B..."):
                df_b = scrape_youtube_comments(url_b, limit_ab)
                if df_b is not None:
                    res_b = []
                    for i, (_, row) in enumerate(df_b.iterrows()):
                        res_b.append(analyze_fallacies(row['content'], api_key=key))
                    df_b = pd.concat([df_b.reset_index(drop=True), pd.DataFrame(res_b)], axis=1)
            
            if df_a is not None and df_b is not None:
                st.divider()
                st.subheader("Match Results")
                c1, c2, c3 = st.columns(3)
                agg_a = df_a['aggression'].mean()
                agg_b = df_b['aggression'].mean()
                delta_agg = agg_a - agg_b
                bots_a = len(df_a[df_a['is_bot']==True])
                bots_b = len(df_b[df_b['is_bot']==True])
                delta_bots = bots_a - bots_b
                fallacy_a = len(df_a[df_a['has_fallacy']==True])
                fallacy_b = len(df_b[df_b['has_fallacy']==True])
                
                c1.metric("Avg Aggression (A vs B)", f"{agg_a:.1f} vs {agg_b:.1f}", f"{delta_agg:.1f}")
                c2.metric("Bot Count (A vs B)", f"{bots_a} vs {bots_b}", f"{delta_bots}")
                c3.metric("Fallacies (A vs B)", f"{fallacy_a} vs {fallacy_b}")
                
                st.subheader("Aggression Comparison")
                df_a['Source'] = 'Dataset A'
                df_b['Source'] = 'Dataset B'
                combined = pd.concat([df_a, df_b])
                chart = alt.Chart(combined).mark_bar().encode(
                    x=alt.X('Source', title=None),
                    y=alt.Y('mean(aggression)', title='Avg Aggression'),
                    color='Source'
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
        else:
            st.error("Please provide both URLs and API Key.")

# ==========================================
# MODULE 3: COGNITIVE EDITOR (LOGIC GUARD)
# ==========================================
elif mode == "Cognitive Editor (Logic Guard)":
    st.header("Cognitive Editor & Fact-Checker")
    st.caption("Paste toxic/fallacious text OR upload an Image (Meme/Screenshot).")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Input")
        inp_type = st.radio("Input Type:", ["Text", "PDF", "Image (Vision Guard)"], horizontal=True)
        
        text_inp = None
        image_inp = None
        
        if inp_type == "Text":
            text_inp = st.text_area("Paste Text Here", height=300)
        elif inp_type == "PDF":
            f = st.file_uploader("Upload PDF", type="pdf")
            if f:
                text_inp = extract_text_from_pdf(f)
                st.success(f"Loaded {len(text_inp)} chars from PDF")
        else:
            img_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
            if img_file:
                image_inp = Image.open(img_file)
                st.image(image_inp, caption="Uploaded Image", use_container_width=True)
            
        if "GEMINI_API_KEY" in st.secrets: k = st.secrets["GEMINI_API_KEY"]
        else: k = st.text_input("API Key", type="password")
        
        go = st.button("Analyze & Sanitize", use_container_width=True)

    with c2:
        st.subheader("Output (Analysis & Sanitize)")
        if go:
            if (text_inp) or (image_inp):
                with st.spinner("Processing with Gemini Vision/Text..."):
                    ret = cognitive_rewrite(text_inp, k, image_inp)
                    if ret:
                        if ret.get('has_fallacy'):
                            st.error(f"🛑 Issue Detected: **{ret['fallacy_type']}**")
                            st.metric("Aggression Level", f"{ret.get('aggression', 0)}/10")
                            st.warning(f"**Analysis:** {ret.get('explanation', 'No details.')}")
                        else:
                            st.success("✅ Neural Guard: No major issues detected.")
                            st.info(f"**Analysis:** {ret.get('explanation', 'Content is sound.')}")
                        
                        st.markdown("---")
                        st.markdown("#### Rewritten Version (Neutral)")
                        st.success(ret.get('rewritten_text', 'No rewrite available.'))
                        st.markdown("---")
                        st.markdown("#### Fact Checker (Claims to Verify)")
                        facts = ret.get('facts', [])
                        if facts:
                            for f in facts:
                                st.write(f"- {f}")
                        else:
                            st.caption("No specific factual claims found.")
            else:
                st.warning("Please provide input.")
