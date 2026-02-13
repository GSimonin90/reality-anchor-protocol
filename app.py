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

# --- SESSION STATE: API COUNTER ---
if 'api_calls' not in st.session_state:
    st.session_state['api_calls'] = 0

def increment_counter():
    st.session_state['api_calls'] += 1

# --- HELPER: ROBUST JSON PARSER ---
def extract_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        return None
    except:
        return None

# --- HELPER: PDF GENERATOR ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'RAP: Deep Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(df, chart_obj=None):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title & Meta
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.ln(5)

    # Statistics
    total = len(df)
    flagged = len(df[df['has_fallacy'] == True])
    clean = total - flagged
    ratio = (flagged / total * 100) if total > 0 else 0
    
    # Sentiment Stats
    avg_aggression = df['aggression'].mean() if 'aggression' in df.columns else 0
    neg_sentiment = len(df[df['sentiment'] == 'Negative']) if 'sentiment' in df.columns else 0

    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Executive Summary", 0, 1)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, f"Total Items Analyzed: {total}", 0, 1)
    pdf.cell(0, 8, f"Logical Issues: {flagged} ({ratio:.1f}%)", 0, 1)
    pdf.cell(0, 8, f"Average Aggression Score: {avg_aggression:.1f}/10", 0, 1)
    pdf.cell(0, 8, f"Negative Sentiment Count: {neg_sentiment}", 0, 1)
    pdf.ln(10)

    # Chart Image
    if chart_obj and VL_CONVERT_AVAILABLE:
        try:
            pdf.set_font("Helvetica", 'B', 14)
            pdf.cell(0, 10, "Fallacy Distribution", 0, 1)
            png_data = vlc.vegalite_to_png(chart_obj.to_json(), scale=2)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(png_data)
                tmp_path = tmp.name
            pdf.image(tmp_path, w=180)
            pdf.ln(10)
            os.unlink(tmp_path)
        except Exception as e:
            pdf.set_font("Helvetica", 'I', 10)
            pdf.cell(0, 10, f"[Chart skipped: {str(e)}]", 0, 1)

    # Detailed Findings
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Critical Findings (Flagged)", 0, 1)
    pdf.ln(5)

    flagged_df = df[df['has_fallacy'] == True]
    
    def sanitize(text):
        clean = str(text).replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
        return clean.encode('latin-1', 'replace').decode('latin-1')

    SAFE_WIDTH = 190 

    for i, row in flagged_df.iterrows():
        pdf.set_font("Helvetica", 'B', 11)
        pdf.set_text_color(200, 0, 0)
        pdf.set_x(10)
        
        # Header with Aggression
        agg_score = row.get('aggression', 0)
        fallacy = sanitize(row.get('fallacy_type', 'Unknown'))
        pdf.cell(0, 8, f"Type: {fallacy} | Aggression: {agg_score}/10", 0, 1)
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", 'I', 10)
        content_snippet = sanitize(row.get('content', ''))
        if len(content_snippet) > 400: content_snippet = content_snippet[:400] + "..."
        pdf.set_x(10)
        pdf.multi_cell(SAFE_WIDTH, 6, f"Text: \"{content_snippet}\"")
        
        pdf.set_font("Helvetica", size=10)
        pdf.set_x(10)
        pdf.multi_cell(SAFE_WIDTH, 6, f"Why: {sanitize(row.get('explanation', ''))}")
        
        pdf.set_text_color(0, 100, 0)
        pdf.set_x(10)
        pdf.multi_cell(SAFE_WIDTH, 6, f"Fix: {sanitize(row.get('correction', ''))}")
        
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

    return bytes(pdf.output())

# --- HELPER: PDF EXTRACTOR ---
@st.cache_data
def extract_text_from_pdf(file_obj):
    try:
        pdf_reader = pypdf.PdfReader(file_obj)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception as e:
        return None

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
                'content': comment.get('text', '')
            })
        if not comments: return None
        return pd.DataFrame(comments)
    except Exception as e:
        return None

# --- HELPER: MOBILE PASTE PARSER ---
def parse_raw_paste(raw_text):
    lines = raw_text.split('\n')
    cleaned_data = []
    noise_triggers = ['reply', 'rispondi', 'responder', 'like', 'mi piace', 'share', 'edited', 'modificato']
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5: continue
        if re.match(r'^\d+[hmwdys]$', line.lower()): continue
        
        is_noise = False
        if line.lower() in noise_triggers: is_noise = True
        
        if not is_noise:
            cleaned_data.append({
                'agent_id': 'Paste_Source',
                'timestamp': 'Unknown',
                'content': line
            })
    return pd.DataFrame(cleaned_data)

# --- HELPER: SMART COLUMN MAPPER ---
def normalize_dataframe(df):
    target_cols = {
        'content': ['content', 'text', 'body', 'tweet', 'tweet_text', 'comment', 'comment_text', 'message', 'caption', 'full_text'],
        'agent_id': ['agent_id', 'author', 'user', 'username', 'handle', 'screen_name', 'from_name', 'owner_username', 'user_id'],
        'timestamp': ['timestamp', 'created_at', 'date', 'time', 'posted_at', 'created_time']
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
            best_col = max(string_cols, key=lambda c: new_df[c].astype(str).str.len().mean())
            new_df = new_df.rename(columns={best_col: 'content'})

    if 'agent_id' not in new_df.columns: new_df['agent_id'] = 'Unknown_User'
    if 'timestamp' not in new_df.columns: new_df['timestamp'] = 'Unknown_Time'
        
    final_cols = ['agent_id', 'timestamp', 'content']
    final_cols = [c for c in final_cols if c in new_df.columns]
    return new_df[final_cols]

# --- HELPER: LOGIC GUARD (CACHED & RETRY) ---
@st.cache_data(show_spinner=False)
def analyze_fallacies_cached(text, api_key):
    if not text or len(str(text)) < 5: return None
    
    for attempt in range(3):
        try:
            client = genai.Client(api_key=api_key)
            is_long_doc = len(text) > 2000
            
            if is_long_doc:
                prompt = f"""
                Analyze LONG DOCUMENT. Identify MAIN fallacy.
                Text snippet: "{text[:15000]}..."
                JSON: {{ 
                    "has_fallacy": true, 
                    "fallacy_type": "Name (English)", 
                    "explanation": "Why (Input Lang)", 
                    "correction": "Fix (Input Lang)",
                    "sentiment": "Positive/Neutral/Negative",
                    "aggression": 1-10 (Integer)
                }}
                """
            else:
                prompt = f"""
                Analyze text for fallacies + sentiment + aggression.
                Text: "{text}"
                JSON: {{ 
                    "has_fallacy": true/false, 
                    "fallacy_type": "Name (English) or None", 
                    "explanation": "Why (Input Lang)", 
                    "correction": "Fix (Input Lang)",
                    "sentiment": "Positive/Neutral/Negative",
                    "aggression": 1-10 (Integer)
                }}
                """

            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            return extract_json(response.text)
            
        except Exception:
            time.sleep(1)
            continue
            
    return {"has_fallacy": False, "fallacy_type": "API Error", "explanation": "Failed", "correction": "", "sentiment": "Neutral", "aggression": 0}

def analyze_fallacies(text, api_key=None):
    if not api_key: return {"has_fallacy": True, "fallacy_type": "No API Key", "explanation": "Key missing.", "correction": "Add key.", "sentiment": "Neutral", "aggression": 0}
    increment_counter()
    return analyze_fallacies_cached(text, api_key)

# --- HEADER ---
st.title("RAP: Reality Anchor Protocol")
st.markdown("### Cognitive Security & Logical Analysis Suite")
st.markdown("---")

mode = st.sidebar.radio("Select Module:", ["Simulation Model", "Social Data Analysis (Universal)", "Logic Guard (Doc/Text)"])
st.sidebar.markdown("---")
st.sidebar.caption(f"API Calls Session: {st.session_state['api_calls']}")

# ==========================================
# MODULE 1: SIMULATION
# ==========================================
if mode == "Simulation Model":
    st.sidebar.header("Parameters")
    n_agents = st.sidebar.slider("Number of Agents", 100, 2000, 1000, 100)
    bot_pct = st.sidebar.slider("Bot Ratio", 0.0, 0.8, 0.40, 0.05)
    steps = st.sidebar.slider("Time Steps", 50, 500, 100, 10)
    rap_active = st.sidebar.checkbox("Activate Reality Anchor Protocol")
    
    if rap_active: st.sidebar.success("System Active")
    else: st.sidebar.warning("System Vulnerable")
    
    # REMOVED BUTTON FOR REACTIVE UPDATES
    # Simulation runs automatically when sliders change
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
# MODULE 2: SOCIAL DATA ANALYSIS
# ==========================================
elif mode == "Social Data Analysis (Universal)":
    st.sidebar.header("Data Source")
    if 'current_df' not in st.session_state: st.session_state['current_df'] = None

    input_method = st.sidebar.radio(
        "Select Input Method:", 
        ["CSV File Upload", "YouTube Link", "Raw Text Paste"],
        horizontal=True
    )
    
    if input_method == "CSV File Upload":
        st.info("Best for Desktop Users with Chrome Extensions.")
        with st.expander("Instructions"):
            st.write("1. Use extensions like 'Instant Data Scraper'.")
            st.write("2. Export data to CSV.")
            st.write("3. Upload below.")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            try:
                raw_df = pd.read_csv(uploaded_file)
                norm_df = normalize_dataframe(raw_df)
                st.session_state['current_df'] = norm_df
                st.success(f"Loaded {len(norm_df)} rows.")
            except: st.error("CSV Error.")

    elif input_method == "YouTube Link":
        st.info("Directly scrape public comments.")
        yt_url = st.text_input("YouTube URL")
        limit = st.slider("Count", 10, 200, 50)
        if st.button("Scrape"):
            with st.spinner("Scraping..."):
                sdf = scrape_youtube_comments(yt_url, limit)
                if sdf is not None:
                    st.session_state['current_df'] = sdf
                    st.success("Scraped!")
                else: st.error("Failed.")

    elif input_method == "Raw Text Paste":
        st.info("Mobile Friendly: Copy comments and paste below.")
        raw_text = st.text_area("Paste content here", height=200)
        if st.button("Process Text Dump"):
            if raw_text:
                parsed_df = parse_raw_paste(raw_text)
                st.session_state['current_df'] = parsed_df
                st.success(f"Extracted {len(parsed_df)} items.")
            else:
                st.error("Text area is empty.")

    df = st.session_state['current_df']
    if df is not None:
        def clean(t): return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', str(t))
        if 'content' in df.columns: df['content'] = df['content'].apply(clean)

        st.markdown("---")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(df.head(50), use_container_width=True)
        with c2:
            st.metric("Items", len(df))
            if len(df) > 0:
                try:
                    text_combined = " ".join(df['content'].astype(str).tolist())
                    wc = WordCloud(width=400, height=200, background_color='black', colormap='Reds').generate(text_combined)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)
                except:
                    st.caption("Not enough text for WordCloud")

        st.markdown("---")
        if "GEMINI_API_KEY" in st.secrets: key = st.secrets["GEMINI_API_KEY"]
        else: key = st.text_input("API Key", type="password")

        c1, c2 = st.columns([1, 3])
        with c1:
            scan_rows = st.slider("Analyze N Rows", 5, 100, 10)
            run = st.button("Run Deep Analysis", disabled=not key)

        if run:
            prog = st.progress(0)
            res = []
            subset = df.head(scan_rows).copy()
            for i, row in subset.iterrows():
                ans = analyze_fallacies(row['content'], api_key=key)
                if ans and "error" not in ans: res.append(ans)
                else: res.append({"has_fallacy": False, "fallacy_type": "Error", "explanation": "Failed", "correction": "", "sentiment": "Neutral", "aggression": 0})
                prog.progress((i+1)/scan_rows)
                time.sleep(0.5)
            
            final_df = pd.concat([subset.reset_index(drop=True), pd.DataFrame(res)], axis=1)
            st.session_state['analyzed_df'] = final_df

        if 'analyzed_df' in st.session_state:
            adf = st.session_state['analyzed_df']
            
            st.markdown("### Analysis Results")
            
            m1, m2, m3 = st.columns(3)
            flagged_count = len(adf[adf['has_fallacy']==True])
            agg_avg = adf['aggression'].mean()
            neg_pct = len(adf[adf['sentiment']=='Negative']) / len(adf) * 100
            
            m1.metric("Logical Fallacies", flagged_count)
            m2.metric("Avg Aggression", f"{agg_avg:.1f}/10")
            m3.metric("Negative Sentiment", f"{neg_pct:.0f}%")

            cnt = adf[adf['has_fallacy']==True]['fallacy_type'].value_counts().reset_index()
            cnt.columns = ['Type', 'Count']
            chart = None
            if not cnt.empty:
                cnt = cnt.sort_values('Count', ascending=False)
                chart = alt.Chart(cnt).mark_bar().encode(
                    x='Count', y=alt.Y('Type', sort='-x'), color=alt.Color('Type', scale=alt.Scale(scheme='magma'))
                ).properties(height=max(300, len(cnt)*35))
                st.altair_chart(chart, use_container_width=True)
            
            c_down1, c_down2 = st.columns([1, 1])
            with c_down1:
                csv = adf.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV Data", csv, "rap_results.csv", "text/csv")
            with c_down2:
                if st.button("Generate PDF Report"):
                    with st.spinner("Rendering PDF..."):
                        pdf_bytes = generate_pdf_report(adf, chart)
                        st.download_button("Download Report (PDF)", pdf_bytes, f"RAP_Report.pdf", "application/pdf")

            st.markdown("---")
            st.subheader("Detailed Feed")
            
            show_flags = st.checkbox("Show Flagged Only", True)
            view = adf[adf['has_fallacy']==True] if show_flags else adf
            
            for _, r in view.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([0.05, 0.95])
                    c1.write("🔴" if r['has_fallacy'] else "🟢")
                    c2.caption(f"{r.get('agent_id', 'User')} | Aggression: {r.get('aggression')}/10 | {r.get('sentiment')}")
                    st.info(r['content'])
                    if r['has_fallacy']:
                        st.markdown(f"**{r['fallacy_type']}**: {r['explanation']}")

# ==========================================
# MODULE 3: DOC GUARD
# ==========================================
elif mode == "Logic Guard (Doc/Text)":
    st.header("Neural Logic Guard")
    c1, c2 = st.columns([2, 1])
    with c1:
        inp_type = st.radio("Input Type:", ["Text", "PDF"], horizontal=True)
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
                st.error(f"FALLACY: {ret['fallacy_type']}")
                st.metric("Aggression Level", f"{ret.get('aggression', 0)}/10")
                st.write(ret['explanation'])
                st.success(ret['correction'])
            else:
                st.success("Logic Sound.")
