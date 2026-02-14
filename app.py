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
from math import pi
from datetime import datetime
from google import genai
import pypdf
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from fpdf import FPDF
from wordcloud import WordCloud

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

if 'data_store' not in st.session_state:
    st.session_state['data_store'] = {
        'CSV File Upload': {'df': None, 'analyzed': None, 'summary': None},
        'YouTube Link': {'df': None, 'analyzed': None, 'summary': None},
        'Raw Text Paste': {'df': None, 'analyzed': None, 'summary': None}
    }

def increment_counter():
    st.session_state['api_calls'] += 1

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
        'counter_reply': ['reply', 'debunk']
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
        "aggression": 0
    }
    
    for key, default_val in defaults.items():
        if key not in data or data[key] is None:
            data[key] = default_val
        if isinstance(data[key], float) and np.isnan(data[key]):
            data[key] = default_val
        if str(data[key]).lower() == 'nan':
            data[key] = default_val

    if data['explanation'] == "Analisi non disponibile." or data['explanation'] == "No analysis provided.":
        if data['has_fallacy']:
            data['explanation'] = f"Rilevata fallacia di tipo: {data['fallacy_type']}."
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
def generate_strategic_summary(df, api_key, context=""):
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
    You are a Strategic Intelligence Analyst. 
    CONTEXT TOPIC: "{context}"
    
    Review data:
    - Avg Aggression: {avg_agg:.1f}/10
    - Trend: {trend_msg}
    - Dominant Emotion: {top_emotion}
    - Bot Activity: {bot_count} items.
    - Fallacies: {fallacy_counts}
    - Topics: {top_topics}
    
    TASK: Write a "Executive Briefing" (max 150 words).
    
    CRITICAL LANGUAGE RULE: 
    - Write in the SAME LANGUAGE as the "CONTEXT TOPIC".
    - If Italian -> Output ITALIAN.
    
    STRUCTURE:
    1. **Overview**: Summary of topic and context gap.
    2. **Dynamics**: Narratives, targets, and emotions.
    3. **Forecast**: Trend assessment (Escalation/Stable).
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

# --- HELPER: HYBRID ANALYZER ---
@st.cache_data(show_spinner=False)
def analyze_fallacies_cached(text, api_key, context_info=""):
    if not text or len(str(text)) < 3: return None
    for attempt in range(3):
        try:
            client = genai.Client(api_key=api_key)
            is_long = len(text) > 2000
            
            # --- MODIFIED PROMPT WITH CATEGORY TRANSLATION INSTRUCTION ---
            common_instructions = f"""
            You are a Logic & Fact Analysis Engine.
            CONTEXT (Global): "{context_info}"
            
            *** CRITICAL INSTRUCTION: LANGUAGE FORCE ***
            1. **DETECT LANGUAGE**: Identify the language of the 'Text'.
            2. **TRANSLATE EVERYTHING**: Translate 'explanation', 'correction', 'counter_reply' AND CATEGORICAL VALUES (Archetype, Emotion) into the DETECTED LANGUAGE.
            3. **FORBIDDEN**: Do NOT output English explanations/labels if the input is Italian/Spanish etc.
            
            TASKS:
            - **explanation**: If Green, write "Reasoning valid" (Translated). If Red, explain WHY.
            - **main_topic**: Central theme (2-3 words, Translated).
            - **target**: Target Entity.
            - **primary_emotion**: Select one: [Anger, Fear, Disgust, Sadness, Joy, Surprise, Neutral] -> TRANSLATE VALUE TO INPUT LANGUAGE (e.g. "Rabbia").
            - **archetype**: Select one: [Instigator, Loyalist, Troll, Rational Skeptic, Observer] -> TRANSLATE VALUE TO INPUT LANGUAGE (e.g. "Istigatore", "Osservatore").
            
            RULES:
            1. **OPINIONS**: Taste/Praise -> 'has_fallacy': false.
            2. **SARCASM**: Sarcasm -> 'has_fallacy': false.
            
            RESPONSE (Strict JSON):
            {{ 
                "has_fallacy": true/false, 
                "fallacy_type": "Translated Name (e.g. Argomento Fantoccio)", 
                "explanation": "MANDATORY: Analysis in INPUT LANGUAGE.", 
                "correction": "Correction in INPUT LANGUAGE (if needed)",
                "main_topic": "Theme (Translated)",
                "target": "Target Entity",
                "primary_emotion": "Emotion (Translated, e.g. Paura)",
                "archetype": "Archetype (Translated, e.g. Istigatore)",
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
            
            if result:
                return sanitize_response(result)
            
        except:
            time.sleep(0.1)
            continue
            
    return sanitize_response(None)

def analyze_fallacies(text, api_key=None, context_info=""):
    if not api_key: return {"has_fallacy": True}
    increment_counter()
    return analyze_fallacies_cached(text, api_key, context_info)

# --- CHARTING HELPERS ---
def plot_radar_chart(df):
    if 'primary_emotion' not in df.columns: return None
    # We take top 7 emotions dynamically to support translated values
    top_emotions = df['primary_emotion'].value_counts().head(7)
    if top_emotions.empty: return None
    
    labels = top_emotions.index.tolist()
    values = top_emotions.values.flatten().tolist()
    
    # Close the loop
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

mode = st.sidebar.radio("Select Module:", ["Simulation Model", "Social Data Analysis (Universal)", "Logic Guard (Doc/Text)"])

# --- DISCLAIMER ---
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ System Capabilities & Limits", expanded=False):
    st.markdown("""
    **What this system CAN do:**
    - Detect logical fallacies (e.g., Ad Hominem, Strawman).
    - Analyze sentiment and aggression levels.
    - Check established facts (History, Science, Geography).
    
    **Blind Spots & Limitations:**
    - **Recent Events:** Knowledge cutoff may exclude news from the last 24-48h.
    - **Niche Topics:** May lack data on obscure local events.
    - **Hallucinations:** AI can occasionally generate incorrect information. 
    
    *Always verify critical claims independently.*
    """)

st.sidebar.markdown("---")
st.sidebar.caption(f"API Calls Session: {st.session_state['api_calls']}")

# ==========================================
# MODULE 1: SIMULATION
# ==========================================
if mode == "Simulation Model":
    st.sidebar.header("Parameters")
    n_agents = st.sidebar.slider("Agents", 100, 2000, 1000)
    bot_pct = st.sidebar.slider("Bot Ratio", 0.0, 0.8, 0.40)
    steps = st.sidebar.slider("Time Steps", 50, 500, 100)
    rap_active = st.sidebar.checkbox("Activate Protocol")
    
    if rap_active: st.sidebar.success("System Active")
    else: st.sidebar.warning("System Vulnerable")
    
    agents = np.zeros(n_agents)
    n_bots = int(n_agents * bot_pct)
    bot_start = int(n_agents * 0.3)
    bot_end = bot_start + n_bots
    agents[bot_start:bot_end] = 1.0
    history = np.zeros((n_agents, steps))
    history[:, 0] = agents.copy()
    current = agents.copy()
    for t in range(1, steps):
        mean = np.mean(current)
        mask = np.ones(n_agents, dtype=bool)
        mask[bot_start:bot_end] = False
        noise = np.random.normal(0, 0.02, n_agents)
        alpha = 0.005 if rap_active else 0.1
        current[mask] += alpha * (mean - current[mask]) + noise[mask]
        current[bot_start:bot_end] = 1.0 
        current = np.clip(current, 0, 1)
        history[:, t] = current.copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(history, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_title(f"State: {'SECURE' if rap_active else 'COMPROMISED'} (Bots: {bot_pct*100:.0f}%)")
    st.pyplot(fig)

# ==========================================
# MODULE 2: SOCIAL DATA
# ==========================================
elif mode == "Social Data Analysis (Universal)":
    st.sidebar.header("Data Source")

    input_method = st.sidebar.radio(
        "Input Method:", 
        ["CSV File Upload", "YouTube Link", "Raw Text Paste"], 
        horizontal=True
    )
    
    # --- GLOBAL CONTEXT INPUT ---
    st.markdown("---")
    context_input = st.text_input("Global Context (Optional)", placeholder="E.g., 'Discussion about Flat Earth', 'Video Title: Politics 2024'")
    st.caption("Provide context to help AI understand sarcasm and specific facts.")
    st.markdown("---")

    current_storage = st.session_state['data_store'][input_method]
    
    if input_method == "CSV File Upload":
        st.info("Desktop: Use 'Instant Data Scraper' extension.")
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
        def clean(t): return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', str(t))
        if 'content' in df.columns: df['content'] = df['content'].apply(clean)

        st.markdown("---")
        c1, c2 = st.columns([2, 1])
        with c1:
            if 'Select' not in df.columns:
                df.insert(0, "Select", False)
            
            edited_df = st.data_editor(
                st.session_state['data_store'][input_method]['df'],
                column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                disabled=["content", "agent_id", "timestamp", "is_bot", "bot_reason", "likes"],
                use_container_width=True,
                hide_index=True,
                key=f"editor_{input_method}",
                height=300
            )
            
            if not edited_df.equals(st.session_state['data_store'][input_method]['df']):
                st.session_state['data_store'][input_method]['df'] = edited_df
                st.rerun()

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
                except:
                    st.caption("Not enough text for WordCloud")

        st.markdown("---")
        if "GEMINI_API_KEY" in st.secrets: key = st.secrets["GEMINI_API_KEY"]
        else: key = st.text_input("API Key", type="password")

        # --- ANALYSIS FORM ---
        with st.form("analysis_exec_form"):
            c_form1, c_form2 = st.columns([1, 4])
            
            selected_rows = edited_df[edited_df.Select]
            has_selection = not selected_rows.empty
            
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
            
            if has_selection:
                subset = selected_rows.copy()
            else:
                subset = df.head(scan_rows).copy()
                
            total_to_scan = len(subset)
            
            for i, (_, row) in enumerate(subset.iterrows()):
                ans = analyze_fallacies(row['content'], api_key=key, context_info=context_input)
                # Fallback handled by sanitize_response inside analyze_fallacies_cached
                res.append(ans)
                
                prog_val = min((i+1)/total_to_scan, 1.0)
                prog.progress(prog_val)
                time.sleep(0.1) 
            
            final_df = pd.concat([subset.reset_index(drop=True), pd.DataFrame(res)], axis=1)
            
            st.session_state['data_store'][input_method]['analyzed'] = final_df
            
            with st.spinner("Generating Strategic Briefing..."):
                summ = generate_strategic_summary(final_df, key, context=context_input)
                st.session_state['data_store'][input_method]['summary'] = summ

        analyzed_df = st.session_state['data_store'][input_method]['analyzed']
        summary_text = st.session_state['data_store'][input_method]['summary']

        if analyzed_df is not None:
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
            
            # --- INTELLIGENCE SUITE VISUALS ---
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
            
            # --- THE ORACLE ---
            st.markdown("---")
            st.subheader("The Oracle (Chat with Data)")
            st.caption("Ask questions about the analyzed data (e.g., 'Who are the main targets?', 'What is the most aggressive argument?').")
            
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

# ==========================================
# MODULE 3: DOC GUARD
# ==========================================
elif mode == "Logic Guard (Doc/Text)":
    st.header("Neural Logic Guard and Fact-Checker")
    c1, c2 = st.columns([2, 1])
    with c1:
        inp_type = st.radio("Input:", ["Text", "PDF"], horizontal=True)
        inp = None
        if inp_type == "Text":
            t = st.text_area("Input", height=150)
            if t: inp = t
        else:
            f = st.file_uploader("PDF", type="pdf")
            if f: inp = extract_text_from_pdf(f)
            if inp: st.success(f"Loaded {len(inp)} chars")

    with c2:
        if "GEMINI_API_KEY" in st.secrets: k = st.secrets["GEMINI_API_KEY"]
        else: k = st.text_input("API Key", type="password")
        go = st.button("Analyze Logic", use_container_width=True)

    if go and inp:
        with st.spinner("Analyzing..."):
            ret = analyze_fallacies(inp, api_key=k)
            if ret.get('has_fallacy'):
                st.error(f"ISSUE: {ret['fallacy_type']}")
                st.metric("Aggression", f"{ret.get('aggression', 0)}/10")
                st.info(f"**Explanation:** {ret['explanation']}")
                st.success(f"**Correction:** {ret['correction']}")
            else:
                st.success("Logic Sound.")
