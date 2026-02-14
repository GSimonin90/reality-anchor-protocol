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

# --- SESSION STATE INITIALIZATION (MULTI-SLOT MEMORY) ---
if 'api_calls' not in st.session_state: st.session_state['api_calls'] = 0

# We create 3 separate slots for the 3 input methods
if 'data_store' not in st.session_state:
    st.session_state['data_store'] = {
        'CSV File Upload': {'df': None, 'analyzed': None, 'summary': None},
        'YouTube Link': {'df': None, 'analyzed': None, 'summary': None},
        'Raw Text Paste': {'df': None, 'analyzed': None, 'summary': None}
    }

def increment_counter():
    st.session_state['api_calls'] += 1

# --- HELPER: ROBUST JSON PARSER ---
def extract_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None
    except:
        return None

# --- HELPER: GLOBAL STRATEGIC SUMMARY ---
def generate_strategic_summary(df, api_key):
    if df is None or df.empty: return None
    
    flagged = df[df['has_fallacy']==True]
    if flagged.empty: return "Analysis shows no logical fallacies or factual errors. The discourse appears healthy and rational."

    fallacy_counts = flagged['fallacy_type'].value_counts().to_string()
    sample_comments = flagged[['content', 'fallacy_type']].head(15).to_string(index=False)
    avg_agg = df['aggression'].mean() if 'aggression' in df.columns else 0
    
    prompt = f"""
    You are a Strategic Intelligence Analyst. 
    Review the following data:
    - Average Aggression: {avg_agg:.1f}/10
    - Fallacy Distribution: {fallacy_counts}
    - Samples: {sample_comments}
    
    TASK: Write a concise "Executive Briefing" (max 150 words) in ENGLISH.
    1. Identify main narrative/attack patterns.
    2. Highlight emotional tone.
    3. Logical vs Factual issues.
    """
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"Could not generate summary: {str(e)}"

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

def generate_pdf_report(df, chart_obj=None, summary_text=None):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.ln(5)

    total = len(df)
    flagged = len(df[df['has_fallacy'] == True])
    ratio = (flagged / total * 100) if total > 0 else 0
    avg_aggression = df['aggression'].mean() if 'aggression' in df.columns else 0
    
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Executive Summary", 0, 1)
    
    if summary_text:
        pdf.set_font("Helvetica", 'I', 11)
        clean_summ = summary_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 6, clean_summ)
        pdf.ln(5)

    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, f"Total Items: {total} | Issues: {flagged} ({ratio:.1f}%)", 0, 1)
    pdf.cell(0, 8, f"Avg Aggression: {avg_aggression:.1f}/10", 0, 1)
    pdf.ln(10)

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
        except: pass

    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Critical Findings Feed", 0, 1)
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
        
        agg_score = row.get('aggression', 0)
        fallacy = sanitize(row.get('fallacy_type', 'Unknown'))
        pdf.cell(0, 8, f"{fallacy} | Aggression: {agg_score}/10", 0, 1)
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", 'I', 10)
        content_snippet = sanitize(row.get('content', ''))
        if len(content_snippet) > 300: content_snippet = content_snippet[:300] + "..."
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
                'content': comment.get('text', '')
            })
        if not comments: return None
        return pd.DataFrame(comments)
    except: return None

# --- HELPER: PASTE PARSER ---
def parse_raw_paste(raw_text):
    lines = raw_text.split('\n')
    cleaned_data = []
    noise = ['reply', 'rispondi', 'responder', 'like', 'mi piace', 'share', 'edited', 'modificato']
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5: continue
        if re.match(r'^\d+[hmwdys]$', line.lower()): continue
        if line.lower() in noise: continue
        cleaned_data.append({'agent_id': 'Paste_Source', 'timestamp': 'Unknown', 'content': line})
    return pd.DataFrame(cleaned_data)

# --- HELPER: DF NORMALIZER ---
def normalize_dataframe(df):
    target_cols = {
        'content': ['content', 'text', 'body', 'tweet', 'tweet_text', 'comment', 'comment_text', 'message'],
        'agent_id': ['agent_id', 'author', 'user', 'username', 'handle', 'owner_username'],
        'timestamp': ['timestamp', 'created_at', 'date', 'time', 'posted_at']
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
    return new_df[['agent_id', 'timestamp', 'content']]

# --- HELPER: HYBRID ANALYZER (SARCASM AWARE + LANGUAGE CONSISTENT) ---
@st.cache_data(show_spinner=False)
def analyze_fallacies_cached(text, api_key):
    if not text or len(str(text)) < 5: return None
    for attempt in range(3):
        try:
            client = genai.Client(api_key=api_key)
            is_long = len(text) > 2000
            
            if is_long:
                prompt = f"""
                Analyze LONG DOCUMENT as Fact-Checker/Logic Analyst.
                1. Identify MAIN FALLACY.
                2. Identify MAIN FACTUAL ERRORS.
                Text: "{text[:15000]}..."
                JSON: {{ "has_fallacy": true, "fallacy_type": "Name", "explanation": "Why", "correction": "Fix", "sentiment": "Neutral", "aggression": 0 }}
                """
            else:
                prompt = f"""
                Analyze text as Rhetoric/Logic/Fact Expert.
                CONTEXT: Social Media often uses SARCASM/IRONY.
                INSTRUCTIONS:
                1. DETECT SARCASM FIRST: If text is sarcastic/satirical, DO NOT flag as Factual Error.
                2. Check REAL LOGICAL FALLACIES.
                3. Check SERIOUS FACTUAL ERRORS.
                
                Text: "{text}"
                
                IMPORTANT: In the 'explanation' field, provide a brief analysis in the SAME LANGUAGE as the Input Text, even if 'has_fallacy' is false (e.g., "Valid reasoning", "Sarcastic comment").

                JSON: {{ 
                    "has_fallacy": true/false, 
                    "fallacy_type": "Name or None", 
                    "explanation": "Explain in INPUT LANGUAGE", 
                    "correction": "Truth in INPUT LANGUAGE",
                    "sentiment": "Positive/Neutral/Negative",
                    "aggression": 1-10
                }}
                """
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            return extract_json(response.text)
        except:
            time.sleep(1)
            continue
    return {"has_fallacy": False, "fallacy_type": "Error", "explanation": "Failed", "correction": "", "sentiment": "Neutral", "aggression": 0}

def analyze_fallacies(text, api_key=None):
    if not api_key: return {"has_fallacy": True}
    increment_counter()
    return analyze_fallacies_cached(text, api_key)

# --- HEADER ---
st.title("RAP: Reality Anchor Protocol")
st.markdown("### Cognitive Security & Logical Analysis Suite")
st.markdown("---")

mode = st.sidebar.radio("Select Module:", ["Simulation Model", "Social Data Analysis (Universal)", "Logic Guard (Doc/Text)"])

# --- RESTORED DETAILED DISCLAIMER ---
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

    # INPUT METHOD SELECTOR
    input_method = st.sidebar.radio(
        "Input Method:", 
        ["CSV File Upload", "YouTube Link", "Raw Text Paste"], 
        horizontal=True
    )
    
    # --- MEMORY RETRIEVAL: Load data specific to this tab ---
    current_storage = st.session_state['data_store'][input_method]
    
    if input_method == "CSV File Upload":
        st.info("Desktop: Use 'Instant Data Scraper' extension.")
        
        # --- RESTORED DETAILED INSTRUCTIONS ---
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

        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            try:
                raw_df = pd.read_csv(uploaded_file)
                norm_df = normalize_dataframe(raw_df)
                # SAVE TO SLOT
                st.session_state['data_store'][input_method]['df'] = norm_df
                # If new file, reset analysis for this slot only
                if current_storage['df'] is not None and not norm_df.equals(current_storage['df']):
                     st.session_state['data_store'][input_method]['analyzed'] = None
                     st.session_state['data_store'][input_method]['summary'] = None
                
                st.success(f"Loaded {len(norm_df)} rows.")
            except: st.error("CSV Error.")

    elif input_method == "YouTube Link":
        yt_url = st.text_input("YouTube URL")
        limit = st.slider("Count", 10, 200, 50)
        if st.button("Scrape"):
            with st.spinner("Scraping..."):
                sdf = scrape_youtube_comments(yt_url, limit)
                if sdf is not None:
                    # SAVE TO SLOT
                    st.session_state['data_store'][input_method]['df'] = sdf
                    st.session_state['data_store'][input_method]['analyzed'] = None # New scrape, reset analysis
                    st.session_state['data_store'][input_method]['summary'] = None
                    st.success("Scraped!")
                else: st.error("Failed.")

    elif input_method == "Raw Text Paste":
        raw_text = st.text_area("Paste content", height=150)
        if st.button("Process"):
            if raw_text:
                pdf_parsed = parse_raw_paste(raw_text)
                # SAVE TO SLOT
                st.session_state['data_store'][input_method]['df'] = pdf_parsed
                st.session_state['data_store'][input_method]['analyzed'] = None # New text, reset analysis
                st.session_state['data_store'][input_method]['summary'] = None
                st.success(f"Extracted {len(pdf_parsed)} items.")

    # --- MAIN LOGIC: READ FROM CURRENT SLOT ---
    df = st.session_state['data_store'][input_method]['df']
    
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

        c1, c2 = st.columns([1, 4])
        with c1:
            scan_rows = st.slider("Rows to Analyze", 5, 100, 10)
            run = st.button("RUN ANALYSIS", disabled=not key, type="primary")

        if run:
            prog = st.progress(0)
            res = []
            subset = df.head(scan_rows).copy()
            for i, row in subset.iterrows():
                ans = analyze_fallacies(row['content'], api_key=key)
                if ans and "error" not in ans: res.append(ans)
                else: res.append({"has_fallacy": False, "fallacy_type": "Error", "explanation": "Failed", "correction": "", "sentiment": "Neutral", "aggression": 0})
                prog.progress((i+1)/scan_rows)
                time.sleep(0.3) 
            
            final_df = pd.concat([subset.reset_index(drop=True), pd.DataFrame(res)], axis=1)
            
            # SAVE ANALYSIS TO SLOT
            st.session_state['data_store'][input_method]['analyzed'] = final_df
            
            with st.spinner("Generating Strategic Briefing..."):
                summ = generate_strategic_summary(final_df, key)
                st.session_state['data_store'][input_method]['summary'] = summ

        # DISPLAY RESULTS FROM SLOT
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
            
            cnt = adf[adf['has_fallacy']==True]['fallacy_type'].value_counts().reset_index()
            cnt.columns = ['Type', 'Count']
            chart = None
            if not cnt.empty:
                chart = alt.Chart(cnt).mark_bar().encode(
                    x='Count', y=alt.Y('Type', sort='-x'), color=alt.Color('Type', scale=alt.Scale(scheme='reds'))
                ).properties(height=max(300, len(cnt)*35))
                st.altair_chart(chart, use_container_width=True)

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
                    c1.write("🔴" if r['has_fallacy'] else "🟢")
                    c2.caption(f"**User:** {r.get('agent_id', 'User')} | **Aggression:** {r.get('aggression')}/10 | **Sentiment:** {r.get('sentiment')}")
                    
                    st.info(f"\"{r['content']}\"")
                    
                    if r['has_fallacy']:
                        st.error(f"**{r['fallacy_type']}**: {r['explanation']}")
                    else:
                        st.success(f"✅ Analysis: {r.get('explanation', 'No issues.')}")

            st.markdown("---")
            c_down1, c_down2 = st.columns([1, 1])
            with c_down1:
                csv = adf.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "rap_data.csv", "text/csv")
            with c_down2:
                if st.button("Generate Full PDF Report"):
                    pdf_bytes = generate_pdf_report(adf, chart, summary_text)
                    st.download_button("Download PDF", pdf_bytes, "RAP_Executive_Report.pdf", "application/pdf")

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
