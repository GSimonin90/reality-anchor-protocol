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
from datetime import datetime, timedelta
from google import genai
from google.genai import types
from PIL import Image, ExifTags
import pypdf
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from fpdf import FPDF
from wordcloud import WordCloud
import streamlit.components.v1 as components
import feedparser
import networkx as nx
import yt_dlp

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RAP Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION ---
if 'api_calls' not in st.session_state: st.session_state['api_calls'] = 0
if 'token_usage' not in st.session_state: st.session_state['token_usage'] = 0
if 'oracle_history' not in st.session_state: st.session_state['oracle_history'] = []
if 'doc_oracle_history' not in st.session_state: st.session_state['doc_oracle_history'] = []
if 'scroll_to_top' not in st.session_state: st.session_state['scroll_to_top'] = False

if 'data_store' not in st.session_state:
    st.session_state['data_store'] = {
        'CSV File Upload': {'df': None, 'analyzed': None, 'summary': None},
        'YouTube Link': {'df': None, 'analyzed': None, 'summary': None},
        'Raw Text Paste': {'df': None, 'analyzed': None, 'summary': None},
        'Telegram Dump (JSON)': {'df': None, 'analyzed': None, 'summary': None},
        'Arena': {'df_a': None, 'df_b': None, 'analyzed_a': None, 'analyzed_b': None},
        'Radar': {'df': None, 'analyzed': None}
    }

def increment_counter(input_text_len=0, output_text_len=0):
    st.session_state['api_calls'] += 1
    in_tokens = input_text_len / 4
    out_tokens = output_text_len / 4
    st.session_state['token_usage'] += (in_tokens + out_tokens)

def get_cost_estimate():
    total_tokens = st.session_state['token_usage']
    cost = (total_tokens / 1_000_000) * 0.20 
    return total_tokens, cost

# --- HELPER: CALLBACK FOR DATA EDITOR ---
def update_editor_state(method_key):
    editor_key = f"editor_{method_key}"
    if editor_key in st.session_state:
        new_data = st.session_state[editor_key]
        if isinstance(new_data, pd.DataFrame):
            st.session_state['data_store'][method_key]['df'] = new_data

# --- HELPER: ROBUST JSON PARSER ---
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
        "explanation": "Analysis not available.",
        "correction": "",
        "main_topic": "General",
        "micro_topic": "General",
        "target": "None",
        "primary_emotion": "Neutral",
        "archetype": "Observer",
        "relevance": "Relevant",
        "counter_reply": "",
        "sentiment": "Neutral",
        "aggression": 0,
        "rewritten_text": "No rewrite necessary.",
        "facts": [],
        "ai_generated_probability": 0,
        "ai_analysis": "AI analysis not performed."
    }
    
    for key, default_val in defaults.items():
        if key not in data or data[key] is None:
            data[key] = default_val
        if isinstance(data[key], float) and np.isnan(data[key]):
            data[key] = default_val
        if str(data[key]).lower() == 'nan':
            data[key] = default_val

    if data['explanation'] in ["Analysis not available.", "Analisi non disponibile.", "No analysis provided."]:
        if data.get('has_fallacy'):
            data['explanation'] = f"Issue detected: {data.get('fallacy_type')}."
        else:
            data['explanation'] = "No issues detected, content appears legitimate."

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

# --- HELPER: TREND PROJECTION ---
@st.cache_data
def project_trend(df, days_ahead=7):
    if 'timestamp' not in df.columns: return None, False
    df_trend = df.copy()
    df_trend['timestamp'] = pd.to_datetime(df_trend['timestamp'], errors='coerce')
    df_trend = df_trend.dropna(subset=['timestamp']).sort_values('timestamp')
    if len(df_trend) < 5: return None, False 
    start_date = df_trend['timestamp'].min()
    df_trend['days_since_start'] = (df_trend['timestamp'] - start_date).dt.days
    x = df_trend['days_since_start'].values
    y = df_trend['aggression'].values
    if len(np.unique(x)) < 2: return None, False 
    slope, intercept = np.polyfit(x, y, 1)
    last_day = x.max()
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1).astype(int)
    forecast_y = slope * future_days + intercept
    future_dates = [start_date + timedelta(days=int(d)) for d in future_days]
    past_df = pd.DataFrame({'Date': df_trend['timestamp'], 'Aggression': y, 'Type': 'Historical'})
    future_df = pd.DataFrame({'Date': future_dates, 'Aggression': forecast_y, 'Type': 'Forecast'})
    combined = pd.concat([past_df, future_df])
    is_escalating = slope > 0.05 
    return combined, is_escalating

# --- HELPER: STRATEGIC SUMMARY ---
def generate_strategic_summary(df, api_key, context="", persona="Intelligence Analyst"):
    if df is None or df.empty: return None
    flagged = df[df['has_fallacy']==True]
    bots = df[df['is_bot']==True]
    fallacy_counts = flagged['fallacy_type'].value_counts().to_string() if not flagged.empty else "None"
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
    CRITICAL LANGUAGE RULE: Write in the SAME LANGUAGE as the "CONTEXT TOPIC".
    """
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        increment_counter(len(prompt), len(response.text))
        return response.text
    except Exception as e:
        return f"Could not generate summary: {str(e)}"

# --- HELPER: ACTION DECK GENERATOR ---
def generate_action_deck(df, action_type, api_key, context):
    if df is None or df.empty: return "No comments selected."
    comments_text = "\n".join([f"- {row['content']} (Issue: {row.get('fallacy_type', 'None')})" for _, row in df.iterrows()])
    prompts = {
        "Generate Counter-Narrative Thread": "Create a Twitter/X thread (3-5 tweets) that politely but firmly debunks the following comments using logic and facts. Tone: Professional but engaging.",
        "Draft Official Statement": "Write a formal Press Release or Official Statement addressing the concerns raised in these comments. Tone: Corporate, reassuring, authoritative.",
        "Legal Risk Assessment": "Analyze these comments for potential defamation, libel, or terms of service violations. Output a bulleted list of actionable legal or moderation steps.",
        "Engagement Strategy": "Suggest 3 specific replies to the most influential comments here to de-escalate the situation and win over the audience."
    }
    prompt = f"""
    CONTEXT: {context}
    ACTION: {action_type}
    INPUT COMMENTS: {comments_text[:15000]}
    INSTRUCTION: {prompts[action_type]}
    OUTPUT LANGUAGE: Same as input comments.
    """
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        increment_counter(len(prompt), len(response.text))
        return response.text
    except Exception as e:
        return f"Error generating action: {str(e)}"

def generate_psyops_profile(df, target_id, api_key):
    target_data = df[df['agent_id'] == target_id]
    if target_data.empty: return "User not found."
    
    comments = "\n".join([f"- {row['content']}" for _, row in target_data.iterrows()])
    
    prompt = f"""
    You are an expert Psychological Operations (Psy-Ops) and Behavioral Analyst.
    Analyze the following messages posted by user '{target_id}':
    {comments[:10000]}
    
    CRITICAL RULE: Detect the language of the messages. You MUST write the ENTIRE response (including all headers, titles, and bullet points) strictly in that SAME language. Do not use English unless the messages are in English.
    
    Provide a "Threat Actor Profile" structured exactly with these 4 points (translated into the target language):
    1. Motivation (Ideological, Troll, Paid Bot, Genuine concern)
    2. Emotional Triggers (What makes them angry?)
    3. Linguistic Profile (Education level, slang, repetitive patterns)
    4. Engagement Recommendation (How to interact with or neutralize them)
    
    Keep it concise and professional.
    """
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        increment_counter(len(prompt), len(response.text))
        return response.text
    except Exception as e:
        return f"Profiling Error: {str(e)}"

# --- HELPER: THE ORACLE (GENERAL) ---
def ask_the_oracle(df, question, api_key, context):
    if df is None: return "No data to analyze."
    subset = df[['agent_id', 'content', 'aggression', 'target', 'main_topic', 'archetype']].head(40).to_string()
    prompt = f"""
    You are "The Oracle", an advanced intelligence system.
    CONTEXT: "{context}"
    DATASET SAMPLE: {subset}
    USER QUESTION: "{question}"
    INSTRUCTIONS: Answer based on data. Be professional. RESPOND IN THE SAME LANGUAGE AS THE USER QUESTION.
    """
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        increment_counter(len(prompt), len(response.text))
        return response.text
    except Exception as e:
        return f"Oracle Error: {str(e)}"

# --- HELPER: DOCUMENT ORACLE (RAG) ---
def ask_document_oracle(full_text, question, api_key):
    prompt = f"""
    You are the "Deep Document Oracle", an AI capable of analyzing massive documents.
    I will provide you with the full text of one or more documents.
    
    USER QUESTION: "{question}"
    
    INSTRUCTIONS:
    - Answer the question comprehensively based ONLY on the provided text.
    - If the text does not contain the answer, say so clearly.
    - Quote specific parts of the text to back up your claims when possible.
    - Respond in the language of the user's question.
    
    DOCUMENT TEXT:
    {full_text}
    """
    try:
        client = genai.Client(api_key=api_key)
        # Using Gemini 2.0 Flash for its massive context window
        response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        increment_counter(len(prompt), len(response.text))
        return response.text
    except Exception as e:
        return f"Document Oracle Error: {str(e)}"

# --- HELPER: EXCEL & PDF ---
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

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'RAP: Strategic Intelligence Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_fallback_chart(df):
    if df is None or 'fallacy_type' not in df.columns: return None
    try:
        counts = df[df['has_fallacy']==True]['fallacy_type'].value_counts()
        if counts.empty: return None
        fig, ax = plt.subplots(figsize=(6, 4))
        counts.head(5).plot(kind='bar', ax=ax, color='firebrick')
        ax.set_title("Top 5 Detected Issues")
        ax.set_ylabel("Count")
        plt.tight_layout()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name)
            return tmp.name
    except: return None

def generate_pdf_report(df, summary_text=None):
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
    chart_path = create_fallback_chart(df)
    if chart_path:
        pdf.image(chart_path, w=170)
        os.unlink(chart_path)
    return bytes(pdf.output())

@st.cache_data(show_spinner=False)
def fetch_youtube_video_bytes(url):
    ydl_opts = {
        'format': 'worst[ext=mp4]/worst',
        'outtmpl': os.path.join(tempfile.gettempdir(), 'yt_temp_vid_%(id)s.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'referer': 'https://www.google.com/',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)
            with open(filepath, 'rb') as f:
                data = f.read()
            os.remove(filepath) # Puliamo le prove
            return data
    except Exception as e:
        return None

# --- HELPER: PDF EXTRACTOR & SCRAPERS ---
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

def parse_raw_paste(raw_text):
    lines = raw_text.split('\n')
    cleaned_data = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5: continue
        cleaned_data.append({'agent_id': 'Paste_Source', 'timestamp': 'Unknown', 'content': line, 'likes': 0})
    return pd.DataFrame(cleaned_data)

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

def parse_telegram_json(file_obj):
    try:
        data = json.load(file_obj)
        messages = data.get('messages', [])
        cleaned = []
        for m in messages:
            if m.get('type') == 'message' and m.get('text'):
                text = m['text']
                if isinstance(text, list): 
                    text = " ".join([t if isinstance(t, str) else t.get('text', '') for t in text])
                if len(text) > 5:
                    cleaned.append({
                        'agent_id': m.get('from', 'Unknown'),
                        'timestamp': m.get('date', 'Unknown'),
                        'content': text,
                        'likes': 0
                    })
        return pd.DataFrame(cleaned)
    except: 
        return None

# --- HELPER: HYBRID ANALYZER ---
@st.cache_data(show_spinner=False)
def analyze_fallacies_cached(text, api_key, context_info="", persona="Logic & Fact Analysis Engine"):
    if not text or len(str(text)) < 3: return sanitize_response(None)
    for attempt in range(3):
        try:
            client = genai.Client(api_key=api_key)
            is_long = len(text) > 2000
            common_instructions = f"""
            You are a {persona}. CONTEXT: "{context_info}"
            CRITICAL: 1. DETECT LANGUAGE. 2. TRANSLATE ALL OUTPUTS TO THAT LANGUAGE.
            TASKS:
            - explanation: Brief Analysis (Translated).
            - main_topic: Central theme (Translated).
            - micro_topic: 1-2 keywords (Translated).
            - target: Target Entity.
            - primary_emotion: [Anger, Fear, Disgust, Sadness, Joy, Surprise, Neutral] (Translated).
            - archetype: [Instigator, Loyalist, Troll, Rational Skeptic, Observer] (Translated).
            RULES: Opinions/Sarcasm -> 'has_fallacy': false.
            RESPONSE (Strict JSON):
            {{ "has_fallacy": bool, "fallacy_type": "Name", "explanation": "Text", "correction": "Text", "main_topic": "Text", "micro_topic": "Text", "target": "Text", "primary_emotion": "Text", "archetype": "Text", "relevance": "Text", "counter_reply": "Text", "sentiment": "Text", "aggression": 0-10 }}
            """
            if is_long: prompt = f"Analyze LONG.\n{common_instructions}\nText: \"{text[:15000]}...\""
            else: prompt = f"Analyze Text.\n{common_instructions}\nText: \"{text}\""
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            result = extract_json(response.text)
            if result: return sanitize_response(result)
        except:
            time.sleep(0.1)
            continue
    return sanitize_response(None)

def analyze_fallacies(text, api_key=None, context_info="", persona="Logic & Fact Analysis Engine"):
    if not api_key: return {"has_fallacy": True}
    res = analyze_fallacies_cached(text, api_key, context_info, persona)
    increment_counter(len(text), 500)
    return res

# --- HELPER: COGNITIVE EDITOR (MULTIMODAL + AI DETECTOR) ---
def cognitive_rewrite(text, api_key, media_data=None, media_type="image"):
    if (not text or len(str(text)) < 3) and not media_data: return None
    try:
        client = genai.Client(api_key=api_key)
        
        prompt_text = f"""
        You are a high-level Strategic Intelligence Investigator, OSINT Specialist, and Digital Forensics Expert. You never provide generic or lazy answers.
        
        YOUR TASKS:
        1. AI FORENSICS: Analyze media for AI GENERATION/ENHANCEMENT. Score 0-20 (Natural), 40-70 (Upscaled/Filtered), 70-100 (Deepfake/Generated). Look for specific artifacts: edge halos, texture loss in skin, unnatural gaze, or "masking" glitches.
        2. IDENTITY VERIFICATION: Identify any famous people, celebrities, or public figures (e.g., 'Tom Cruise'). Evaluate if their appearance, age, and movements are consistent with known records. State clearly if the subject is a known individual being impersonated via Deepfake.
        3. ADVANCED GEO-INT (Shadow Geolocation): MANDATORY: Deduce the geographic location by identifying micro-clues. Analyze: power outlets/plugs, street signs, architectural styles, car license plates, specific vegetation, logos (e.g., 'Barone Firenze'), or language on background objects. Aim for city or region level.
        4. SYLLOGISM MACHINE: If text is provided, deconstruct the core argument into formal logical steps (Premise 1, Premise 2, Conclusion).
        5. VIDEO TIMELINE: MANDATORY: If the media is a video, you MUST provide at least 5-8 timestamp objects in the 'video_timeline' array. Identify the exact moments where AI artifacts or facial "masking" become more prominent with technical details.
        6. AGGRESSION: Score the emotional intensity/aggression from 0 to 10 (MANDATORY RANGE: 0-10).
        
        CRITICAL LANGUAGE RULE: 
        1. IF 'Input Text/Context' IS PROVIDED: Detect its language and use it for ALL output values.
        2. IF 'Input Text/Context' IS EMPTY: Check the MEDIA content. If it is English, use ENGLISH. If it is Italian, use ITALIAN. 
        3. DEFAULT FALLBACK: If unsure, use ENGLISH. Never use German or other languages.
        4. ABSOLUTE CONSISTENCY: Do not mix languages. If English is detected, every single field (explanation, details, ai_analysis, rewritten_text) MUST be in English.
        
        JSON OUTPUT RULES (Keep keys in English):
        - "fallacy_type": Name of the issue/fallacy in the target language.
        - "explanation": Comprehensive analysis in the target language.
        - "ai_analysis": Detailed forensic breakdown (explicitly mention identified people and brands) in the target language.
        - "syllogism_breakdown": Array of objects with 'step', 'text', 'flaw' in the target language.
        - "video_timeline": Array of objects (MANDATORY) with 'timestamp', 'ai_score', 'details' (details in the target language).
        - "shadow_geolocation": String with detailed geographic deduction (mention specific clues found) in the target language.
        - "aggression": Integer (STRICTLY 0 to 10).
        
        RESPONSE (Strict JSON):
        {{
            "has_fallacy": true/false,
            "fallacy_type": "",
            "explanation": "",
            "rewritten_text": "",
            "facts": [],
            "aggression": 0,
            "ai_generated_probability": 0,
            "ai_analysis": "",
            "voice_stress_score": 0,
            "shadow_geolocation": "",
            "syllogism_breakdown": [],
            "video_timeline": [],
            "search_sources": []
        }}
        """
        
        contents = [prompt_text]
        if text: contents.append(f"Input Text/Context: {text[:10000]}")
        
        if media_data:
            if media_type == "image":
                contents.append(media_data) 
            elif media_type == "audio":
                contents.append(types.Part.from_bytes(data=media_data, mime_type="audio/mp3"))
            elif media_type == "video":
                contents.append(types.Part.from_bytes(data=media_data, mime_type="video/mp4"))

        config_tools = types.GenerateContentConfig(tools=[{"google_search": {}}])
        response = client.models.generate_content(model='gemini-2.0-flash', contents=contents, config=config_tools)
        increment_counter(len(str(contents)), len(response.text))
        res = extract_json(response.text)
        return sanitize_response(res)
    except Exception as e:
        return {"explanation": f"Error processing media: {str(e)}", "has_fallacy": True}

# --- VISUALIZATION HELPERS ---
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

def plot_network_graph(df):
    if 'target' not in df.columns or 'micro_topic' not in df.columns: return None
    valid = df[(df['target'] != "Unknown") & (df['target'] != "None") & (df['micro_topic'] != "Unknown")]
    if valid.empty: return None
    
    G = nx.Graph()
    for _, row in valid.iterrows():
        target = f"{row['target']}"
        topic = f"{row['micro_topic']}"
        G.add_edge(target, topic, weight=1)
        
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', 
            node_size=2000, font_size=9, font_weight='bold', ax=ax)
    ax.set_title("Entity-Narrative Network", size=14)
    return fig

def plot_document_entity_graph(json_data):
    if not json_data or 'relations' not in json_data: return None
    
    G = nx.Graph()
    for rel in json_data['relations']:
        src = str(rel.get('source', '')).strip()
        tgt = str(rel.get('target', '')).strip()
        label = str(rel.get('relation', '')).strip()
        if src and tgt:
            G.add_edge(src, tgt, label=label)
            
    if len(G.edges) == 0: return None
            
    fig, ax = plt.subplots(figsize=(12, 9))
    pos = nx.spring_layout(G, k=1.5, seed=42)
    
    nx.draw(G, pos, with_labels=True, node_color='#ff7f0e', edge_color='#d3d3d3', 
            node_size=2500, font_size=9, font_weight='bold', ax=ax, alpha=0.9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    
    texts = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8,
                                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=0.3))
    
    for _, t in texts.items():
        t.set_zorder(10)
    
    ax.set_title("Secret Document Power Graph", size=14)
    return fig

# --- MAIN UI ---
st.title("RAP: Reality Anchor Protocol")
st.markdown("### Cognitive Security & Logical Analysis Suite")
st.markdown("---")

# ORDERED MODULES
mode = st.sidebar.radio("Select Module:", [
    "1. Wargame Room (Simulation)", 
    "2. Social Data Analysis (Universal)", 
    "3. Cognitive Editor (Text/Image/Audio)", 
    "4. Comparison Test (A/B Testing)",
    "5. Live Radar (RSS/Reddit)",
    "6. Deep Document Oracle (RAG)"
])

# --- DISCLAIMER & METRICS ---
st.sidebar.markdown("---")
if "GEMINI_API_KEY" in st.secrets:
    key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("API Key loaded securely")
else:
    key = st.sidebar.text_input("API Key", type="password")

toks, cost = get_cost_estimate()
st.sidebar.caption(f"Session Usage: ~{int(toks)} tokens")
st.sidebar.caption(f"Est. Cost: ${cost:.4f}")

with st.sidebar.expander("ℹ️ Capabilities", expanded=False):
    st.markdown("""
    **System Capabilities:**
    - **Text/Social:** Analysis of fallacies, bots, and trends.
    - **Vision & EXIF:** Analysis of memes, deepfakes, and hidden metadata.
    - **Audio:** Tone & logic analysis of speech.
    - **Radar:** Live monitoring of RSS feeds and Reddit.
    - **Oracle:** Deep chat with 1M+ token context for massive PDFs.
    """)

st.sidebar.markdown("---")

# ==========================================
# MODULE 1: WARGAME ROOM (SIMULATION)
# ==========================================
if mode == "1. Wargame Room (Simulation)":
    st.header("1. Information Warfare Simulator")
    
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
    
    if topology == "Public Square (High Connectivity)": influence_strength, noise_level = 0.05, 0.02
    elif topology == "Echo Chambers (Clusters)": influence_strength, noise_level = 0.15, 0.01
    else: influence_strength, noise_level = 0.03, 0.01
        
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
elif mode == "2. Social Data Analysis (Universal)":
    st.header("2. Social Data Analysis")
    st.sidebar.header("Settings")
    persona = st.sidebar.selectbox("Analysis Lens (Persona)", ["Strategic Intelligence Analyst", "Mass Psychologist (Emotional)", "Legal Consultant (Defamation/Risk)", "Campaign Manager (Opportunity)"])
    
    input_method = st.sidebar.radio("Input Method:", ["CSV File Upload", "YouTube Link", "Raw Text Paste", "Telegram Dump (JSON)"], horizontal=True)
    st.markdown("---")
    context_input = st.text_input("Global Context (Optional)", placeholder="E.g., 'Discussion about Flat Earth'")
    st.markdown("---")

    current_storage = st.session_state['data_store'][input_method]
    
    if input_method == "CSV File Upload":
        st.info("Desktop: Use 'Instant Data Scraper' extension.")
        with st.expander("📝 How to extract data from Facebook, X (Twitter), Instagram"):
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

    elif input_method == "Telegram Dump (JSON)":
        with st.form("tg_form"):
            tg_file = st.file_uploader("Upload Telegram Chat Export (JSON)", type="json")
            submitted = st.form_submit_button("Extract Intel")
            if submitted and tg_file:
                with st.spinner("Decrypting Telegram Dump..."):
                    tg_df = parse_telegram_json(tg_file)
                    if tg_df is not None:
                        tg_df = detect_bot_activity(tg_df)
                        st.session_state['data_store'][input_method]['df'] = tg_df
                        st.session_state['data_store'][input_method]['analyzed'] = None 
                        st.session_state['data_store'][input_method]['summary'] = None
                        st.success(f"Intercepted {len(tg_df)} messages.")
                    else:
                        st.error("Invalid Telegram JSON format.")

    df = st.session_state['data_store'][input_method]['df']
    
    if df is not None:
        if isinstance(df, pd.DataFrame):
            def clean(t): return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', str(t))
            if 'content' in df.columns: df['content'] = df['content'].apply(clean)

            st.markdown("---")
            c1, c2 = st.columns([2, 1])
            with c1:
                if 'Select' not in df.columns: df.insert(0, "Select", False)
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
                            from wordcloud import STOPWORDS
                            
                            # --- MODIFICA WORDCLOUD: Filtro Stop-Words ITA/ENG ---
                            ita_stops = {
                                "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da", "in", "con", "su", "per", "tra", "fra", 
                                "e", "o", "ma", "se", "perché", "non", "che", "chi", "cui", "mi", "ti", "ci", "vi", "si", "ho", "ha", "hanno", 
                                "è", "sono", "sei", "siamo", "siete", "era", "erano", "c'è", "ne", "al", "allo", "alla", "ai", "agli", "alle", 
                                "del", "dello", "della", "dei", "degli", "delle", "dal", "dallo", "dalla", "dai", "dagli", "dalle", "nel", "nello", 
                                "nella", "nei", "negli", "nelle", "sul", "sullo", "sulla", "sui", "sugli", "sulle", "questo", "quello", "più", 
                                "anche", "tutto", "tutti", "solo", "fare", "fatto", "essere", "stato", "poi", "quando", "molto", "così", "quindi", 
                                "dopo", "invece", "ancora", "già", "senza", "sempre", "ora", "qui", "lì", "quale", "cosa", "loro", "come"
                            }
                            # Uniamo il filtro inglese di default con la nostra black-list italiana
                            custom_stops = set(STOPWORDS).union(ita_stops)
                            
                            # Convertiamo il testo in minuscolo per evitare duplicati come "Governo" e "governo"
                            text_combined = " ".join(df['content'].astype(str).tolist()).lower()
                            
                            wc = WordCloud(width=400, height=200, background_color='black', colormap='Reds', random_state=42, stopwords=custom_stops).generate(text_combined)
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
                        if scan_rows > 100: st.warning("⚠️ High volume analysis")
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
                
                st.session_state['scroll_to_top'] = True
                st.rerun()

            analyzed_df = st.session_state['data_store'][input_method]['analyzed']
            summary_text = st.session_state['data_store'][input_method]['summary']

            if analyzed_df is not None:
                st.markdown('<div id="briefing_anchor"></div>', unsafe_allow_html=True)
                if st.session_state.get('scroll_to_top'):
                    components.html(
                        """<script>setTimeout(function() {var element = window.parent.document.getElementById("briefing_anchor");if (element) {element.scrollIntoView({behavior: "smooth", block: "start"});}}, 800);</script>""",
                        height=0
                    )
                    st.session_state['scroll_to_top'] = False

                adf = analyzed_df
                if summary_text:
                    st.markdown("### Strategic Intelligence Briefing")
                    st.info(summary_text)
                
                m1, m2, m3 = st.columns(3)
                flagged_count = len(adf[adf['has_fallacy']==True])
                agg_avg = adf['aggression'].mean()
                m1.metric("Issues Detected", flagged_count)
                m2.metric("Avg Aggression", f"{agg_avg:.1f}/10")
                
                # --- ACTION DECK ---
                st.markdown("---")
                st.subheader("Response Strategy & Action Deck")
                st.caption("1. Select comments below. 2. Choose an action. 3. Generate.")
                
                if 'Select' not in adf.columns: adf.insert(0, "Select", False)
                action_view = adf[['Select', 'content', 'fallacy_type', 'aggression']].copy()
                edited_action_df = st.data_editor(
                    action_view,
                    column_config={
                        "Select": st.column_config.CheckboxColumn(required=True),
                        "content": st.column_config.TextColumn("Comment Content", width="large"),
                        "fallacy_type": st.column_config.TextColumn("Issue"),
                        "aggression": st.column_config.NumberColumn("Agg", width="small")
                    },
                    key="action_selector_table", height=200, hide_index=True
                )

                action_col1, action_col2 = st.columns([3, 1])
                with action_col1:
                    action_type = st.selectbox("Select Action Type", [
                        "Generate Counter-Narrative Thread", 
                        "Draft Official Statement", 
                        "Legal Risk Assessment", 
                        "Engagement Strategy"
                    ])
                with action_col2:
                    st.write("") 
                    st.write("") 
                    gen_action = st.button("Generate Strategy", type="primary")
                
                if gen_action:
                    selected_for_action = edited_action_df[edited_action_df.Select]
                    if not selected_for_action.empty:
                        with st.spinner("Generating..."):
                            action_plan = generate_action_deck(selected_for_action, action_type, key, context_input)
                            st.success("Action Plan Ready:")
                            st.text_area("Strategy Output:", value=action_plan, height=300)
                            st.download_button("Download Strategy", action_plan, "strategy.txt")
                    else: st.warning("Please select at least one comment in the table above.")

                st.markdown("---")
                st.markdown("---")
                st.subheader("Psy-Ops Target Profiler")
                st.caption("Select a specific User/Agent to run a deep behavioral analysis on their messaging patterns.")
                
                c_psy1, c_psy2 = st.columns([1, 2])
                with c_psy1:
                    unique_users = adf['agent_id'].unique()
                    selected_target = st.selectbox("Select Target (Agent ID)", unique_users)
                    run_psyops = st.button("Run Behavioral Profile", type="secondary")
                
                with c_psy2:
                    if run_psyops:
                        with st.spinner(f"Profiling {selected_target}..."):
                            profile_res = generate_psyops_profile(adf, selected_target, key)
                            st.success(f"Profile Generated for: {selected_target}")
                            st.markdown(profile_res)
                st.markdown("---")
                with st.expander("📊 Open Intelligence Visuals (Radar, Heatmap & Targets)", expanded=False):
                    c_vis1, c_vis2 = st.columns(2)
                    with c_vis1:
                        radar_fig = plot_radar_chart(adf)
                        if radar_fig: st.pyplot(radar_fig)
                    with c_vis2:
                        heatmap = plot_heatmap(adf)
                        if heatmap: st.altair_chart(heatmap, use_container_width=True)
                    st.markdown("---")
                    c_intel1, c_intel2 = st.columns(2)
                    with c_intel1:
                        st.caption("Emerging Narratives (Micro-Topics)")
                        if 'micro_topic' in adf.columns:
                            top_clusters = adf['micro_topic'].value_counts().reset_index().head(10)
                            top_clusters.columns = ['Cluster', 'Count']
                            chart_cluster = alt.Chart(top_clusters).mark_bar().encode(
                                x='Count', y=alt.Y('Cluster', sort='-x'), color=alt.Color('Cluster', legend=None)
                            ).properties(height=300)
                            st.altair_chart(chart_cluster, use_container_width=True)
                    with c_intel2:
                        st.caption("User Archetypes")
                        if 'archetype' in adf.columns:
                            top_arch = adf['archetype'].value_counts().reset_index()
                            top_arch.columns = ['Archetype', 'Count']
                            chart_arch = alt.Chart(top_arch).mark_bar().encode(
                                x='Count', y=alt.Y('Archetype', sort='-x'), color=alt.Color('Archetype', scale=alt.Scale(scheme='dark2'), legend=None)
                            ).properties(height=300)
                            st.altair_chart(chart_arch, use_container_width=True)
                with st.expander("Target-Narrative Network Graph", expanded=False):
                    net_fig = plot_network_graph(adf)
                    if net_fig:
                        st.pyplot(net_fig)
                    else:
                        st.caption("Not enough Target/Topic data to build a network.")

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
                    st.caption("Crisis Trend (Forecast)")
                    proj_data, is_escalating = project_trend(adf)
                    if proj_data is not None:
                        base = alt.Chart(proj_data).mark_line().encode(
                            x=alt.X('Date:T', title='Timeline'),
                            y=alt.Y('Aggression', title='Aggression (0-10)'),
                            color=alt.Color('Type', scale=alt.Scale(domain=['Historical', 'Forecast'], range=['orange', 'red'])),
                            tooltip=['Date:T', 'Aggression', 'Type']
                        )
                        st.altair_chart(base, use_container_width=True)
                        st.caption(f"**Forecast:** {'⚠️ Escalating' if is_escalating else '📉 Stabilizing'}")
                    else:
                        st.caption("Sequence Trend (No valid dates found)")
                        adf['Sequence'] = adf.index
                        line = alt.Chart(adf).mark_line(color='orange').encode(x='Sequence', y='aggression', tooltip=['content', 'aggression'])
                        st.altair_chart(line, use_container_width=True)

                st.markdown("---")
                st.subheader("Detailed Data Explorer")
                c_filter1, c_filter2, c_filter3 = st.columns(3)
                with c_filter1:
                    filter_fallacy = st.multiselect("Filter by Issue Type", adf['fallacy_type'].unique())
                with c_filter2:
                    filter_topic = []
                    if 'micro_topic' in adf.columns:
                        filter_topic = st.multiselect("Filter by Narrative", adf['micro_topic'].unique())
                with c_filter3:
                    min_agg = st.slider("Min Aggression", 0, 10, 0)
                
                view = adf.copy()
                if filter_fallacy: view = view[view['fallacy_type'].isin(filter_fallacy)]
                if filter_topic and 'micro_topic' in view.columns: view = view[view['micro_topic'].isin(filter_topic)]
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
                        pdf_bytes = generate_pdf_report(adf, summary_text=summary_text) 
                        st.download_button("Download PDF", pdf_bytes, "RAP_Executive_Report.pdf", "application/pdf")
                
                st.markdown("---")
                st.subheader("The Oracle (Chat with Data)")
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
# MODULE 3: COGNITIVE EDITOR (MULTIMODAL)
# ==========================================
elif mode == "3. Cognitive Editor (Text/Image/Audio)":
    st.header("3. Cognitive Editor & Fact-Checker")
    st.caption("Upload Text, Images (Memes/Screenshots), Audio clips or Video/Youtube links for deep inspection.")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Input")
        inp_type = st.radio("Input Type:", ["Text", "PDF", "Image (Vision Guard)", "Audio (Voice Intel)", "Video (Deepfake Scan)"], horizontal=True)
        
        text_inp = None
        media_inp = None
        media_type = "text"
        exif_data = {}
        
        if inp_type == "Text":
            text_inp = st.text_area("Paste Text Here", height=300)
        elif inp_type == "PDF":
            f = st.file_uploader("Upload PDF", type="pdf")
            if f:
                text_inp = extract_text_from_pdf(f)
                st.success(f"Loaded {len(text_inp)} chars from PDF")
        elif inp_type == "Image (Vision Guard)":
            f = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
            text_inp = st.text_area("Image Context / Post Text (Optional)", placeholder="E.g., Paste the Facebook/X post text that accompanied this photo...", height=100)
            if f:
                media_inp = Image.open(f)
                media_type = "image"
                st.image(media_inp, caption="Uploaded Image", use_container_width=True)
                
                # --- EXIF OSINT EXTRACTION & GPS GEOLOCATION ---
                gps_coords = None
                try:
                    exif_info = media_inp._getexif()
                    if exif_info:
                        if 34853 in exif_info:
                            gps_info = {ExifTags.GPSTAGS.get(k, k): v for k, v in exif_info[34853].items()}
                            
                            if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                                lat_d = gps_info['GPSLatitude']
                                lon_d = gps_info['GPSLongitude']
                                lat_ref = gps_info.get('GPSLatitudeRef', 'N')
                                lon_ref = gps_info.get('GPSLongitudeRef', 'E')
                                
                                lat = float(lat_d[0]) + float(lat_d[1])/60 + float(lat_d[2])/3600
                                if lat_ref == 'S': lat = -lat
                                
                                lon = float(lon_d[0]) + float(lon_d[1])/60 + float(lon_d[2])/3600
                                if lon_ref == 'W': lon = -lon
                                
                                gps_coords = pd.DataFrame({'lat': [lat], 'lon': [lon]})

                        for tag_id, value in exif_info.items():
                            tag = ExifTags.TAGS.get(tag_id, tag_id)
                            if tag != "MakerNote":
                                exif_data[tag] = str(value)
                except Exception as e:
                    pass
                
                if gps_coords is not None:
                    st.success("🌍 **Location Traces Detected (GPS):** Geolocation data found!")
                    st.map(gps_coords, zoom=12, use_container_width=True)
                
                if exif_data:
                    software_used = exif_data.get('Software') or exif_data.get('ProcessingSoftware')
                    date_original = exif_data.get('DateTimeOriginal') or exif_data.get('DateTime')
                    
                    with st.expander("Invisible EXIF Metadata Found (OSINT)", expanded=True):
                        
                        if date_original:
                            st.info(f"**Original Creation Date:** {date_original}")
                        else:
                            st.caption("**Creation Date:** Not found in metadata.")
                            
                        if software_used:
                            st.warning(f"**Editing Trace Detected:** The image was modified using **{software_used}**.")
                        else:
                            st.success("✅ **Clean Metadata:** No image editing software detected in EXIF.")
                            
                        st.json(exif_data)
                else:
                    st.caption("No EXIF data found (Image might be scrubbed by social media).")

        elif inp_type == "Audio (Voice Intel)":
            f = st.file_uploader("Upload Audio", type=['mp3', 'wav', 'm4a'])
            if f:
                media_inp = f.read() # Read bytes
                media_type = "audio"
                st.audio(media_inp, format='audio/mp3')
        
        elif inp_type == "Video (Deepfake Scan)":
            v_mode = st.radio("Video Source:", ["Upload File (MP4/MOV)", "YouTube Link"], horizontal=True)
            text_inp = st.text_area("Video Context (Optional)", placeholder="What is this video claiming?", height=100)
            
            yt_url_input = None 
            
            if v_mode == "Upload File (MP4/MOV)":
                f = st.file_uploader("Upload Video (Max 50MB)", type=['mp4', 'mov'])
                if f:
                    media_inp = f.read()
                    media_type = "video"
                    st.video(media_inp)
            else:
                yt_url_input = st.text_input("Paste YouTube URL here")
                if yt_url_input:
                    st.video(yt_url_input) 
                    media_type = "video"
            
        go = st.button("Analyze, Sanitize & Scan AI", use_container_width=True, type="primary")

    with c2:
        st.subheader("Output (Analysis & Sanitize)")
        if go:
            # --- BACKGROUND YOUTUBE DOWNLOAD SECTION ---
            if inp_type == "Video (Deepfake Scan)" and yt_url_input:
                with st.spinner("Downloading YouTube video in background for analysis (optimized quality)..."):
                    media_inp = fetch_youtube_video_bytes(yt_url_input)
                    if not media_inp:
                        st.error("Failed to download the video. It might be private, age-restricted, or no longer available.")
                        st.stop()
            # ----------------------------------------------
                        
            if (text_inp) or (media_inp):
                with st.spinner(f"Processing with Gemini ({inp_type})..."):
                    ret = cognitive_rewrite(text_inp, key, media_inp, media_type)
                    
                    if ret:
                        # --- AI SCANNER UI ---
                        ai_prob = ret.get('ai_generated_probability', 0)
                        if ai_prob > 75:
                            st.error(f"🤖 **HIGH PROBABILITY OF AI GENERATION: {ai_prob}%**")
                            st.caption(ret.get('ai_analysis', 'Detected deepfake/LLM patterns.'))
                        elif ai_prob > 40:
                            st.warning(f"⚠️ **SUSPICIOUS AI GENERATION SCORE: {ai_prob}%**")
                            st.caption(ret.get('ai_analysis', 'Possible use of AI tools.'))
                        else:
                            st.success(f"👤 **LIKELY HUMAN GENERATED (AI Score: {ai_prob}%)**")
                        
                        st.markdown("---")

                        # --- FORENSIC VIDEO TIMELINE ---
                        v_timeline = ret.get('video_timeline', [])
                        if media_type == "video" and isinstance(v_timeline, list) and len(v_timeline) > 0:
                            st.markdown("#### Forensic Video Timeline")
                            st.caption("Temporal analysis of AI manipulation probability across the video length.")
                            
                            vt_df = pd.DataFrame(v_timeline)
                            
                            rename_map = {}
                            for col in vt_df.columns:
                                if col.lower() in ['time', 'timestamp', 'minuti']: rename_map[col] = 'timestamp'
                                if col.lower() in ['ai_score', 'score', 'probabilità', 'probability']: rename_map[col] = 'ai_score'
                                if col.lower() in ['details', 'dettagli', 'info']: rename_map[col] = 'details'
                            vt_df = vt_df.rename(columns=rename_map)

                            if 'timestamp' in vt_df.columns and 'ai_score' in vt_df.columns:
                                vt_df['ai_score'] = pd.to_numeric(vt_df['ai_score'], errors='coerce').fillna(0)
                                
                                chart_vt = alt.Chart(vt_df).mark_line(point=True, color='red').encode(
                                    x=alt.X('timestamp:O', title='Timestamp'),
                                    y=alt.Y('ai_score:Q', title='AI Probability (%)', scale=alt.Scale(domain=[0, 100])),
                                    tooltip=['timestamp', 'ai_score', 'details']
                                ).properties(height=250)
                                st.altair_chart(chart_vt, use_container_width=True)
                                
                                with st.expander("View Frame-by-Frame Details"):
                                    for _, row in vt_df.iterrows():
                                        st.write(f"**{row.get('timestamp', '??:??')}** (Score: {int(row.get('ai_score', 0))}%) - {row.get('details', '')}")
                            else:
                                st.warning("Timeline data format inconsistent. Check RAW output.")
                            st.markdown("---")
                        # ---------------------------------------------

                        # --- FALLACY & LOGIC UI ---
                        if ret.get('has_fallacy'):
                            st.error(f"🛑 Issue Detected: **{ret['fallacy_type']}**")
                            st.metric("Aggression Level", f"{ret.get('aggression', 0)}/10")
                            st.warning(f"**Analysis:** {ret.get('explanation', 'No details.')}")
                        else:
                            st.success("✅ Neural Guard: No major issues detected.")
                            st.info(f"**Analysis:** {ret.get('explanation', 'Content is sound.')}")
                        
                        st.markdown("---")
                        syl_breakdown = ret.get('syllogism_breakdown', [])
                        if syl_breakdown and len(syl_breakdown) > 0:
                            st.markdown("#### Decostruzione Logica")
                            st.caption("Il testo è stato frammentato in premesse formali per individuare il punto esatto in cui la logica fallisce.")
                            for step in syl_breakdown:
                                flaw_text = str(step.get('flaw', '')).strip()
                                step_name = step.get('step', 'Step')
                                text_val = step.get('text', '')
                                
                                if flaw_text and flaw_text.lower() not in ["none", "nessuno", "nessuna", "n/a", "", "null"]:
                                    st.error(f"**{step_name}**: {text_val}\n\n⚠️ **Salto Logico / Fallacia:** {flaw_text}")
                                else:
                                    st.info(f"**{step_name}**: {text_val}")
                            st.markdown("---")
                        # -------------------------------------------
                        st.markdown("#### Rewritten Version / Transcript Summary")
                        st.info(ret.get('rewritten_text', 'No rewrite available.'))
                        st.markdown("---")
                        if media_type == "audio" and 'voice_stress_score' in ret:
                            stress = ret.get('voice_stress_score', 0)
                            st.markdown("####Voice & Tone Analysis")
                            if stress > 65:
                                st.error(f"**Voice Stress Score: {stress}%** (High emotion, anger, or panic detected in prosody)")
                            else:
                                st.success(f"**Voice Stress Score: {stress}%** (Calm, controlled, or neutral tone)")
                            st.markdown("---")
                        if media_type in ["image", "video"] and ret.get('shadow_geolocation') and ret.get('shadow_geolocation') != "Not applicable":
                            st.markdown("#### Shadow Geolocation (Visual GeoINT)")
                            st.info(f"**AI Visual Deduction:** {ret['shadow_geolocation']}")
                            st.markdown("---")
                        st.markdown("#### Fact Checker (Claims to Verify)")
                        facts = ret.get('facts', [])
                        if facts:
                            for f in facts: st.write(f"- {f}")
                        else:
                            st.caption("No specific factual claims found.")
            else:
                st.warning("Please provide input.")

# ==========================================
# MODULE 4: COMPARISON TEST (A/B TESTING)
# ==========================================
elif mode == "4. Comparison Test (A/B Testing)":
    st.header("4. Comparison Test (A/B Testing)")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Dataset A (Left)")
        url_a = st.text_input("YouTube URL A", key="url_a")
    with col_b:
        st.subheader("Dataset B (Right)")
        url_b = st.text_input("YouTube URL B", key="url_b")
    
    limit_scrape = st.number_input("Raw Comments to Fetch", 10, 1000, 50, help="How many comments to download initially.")
    
    if "GEMINI_API_KEY" in st.secrets: key = st.secrets["GEMINI_API_KEY"]
    else: key = st.text_input("API Key", type="password")

    if st.button("Step 1: Fetch Comments (Scrape Only)", type="primary"):
        if url_a and url_b:
            with st.spinner("Fetching raw data from both videos..."):
                df_a = scrape_youtube_comments(url_a, limit_scrape)
                if df_a is not None:
                    df_a = detect_bot_activity(df_a)
                    df_a.insert(0, "Select", False)
                    st.session_state['data_store']['Arena']['df_a'] = df_a
                
                df_b = scrape_youtube_comments(url_b, limit_scrape)
                if df_b is not None:
                    df_b = detect_bot_activity(df_b)
                    df_b.insert(0, "Select", False)
                    st.session_state['data_store']['Arena']['df_b'] = df_b
                
                st.session_state['data_store']['Arena']['analyzed_a'] = None
                st.session_state['data_store']['Arena']['analyzed_b'] = None
                st.rerun()
        else:
            st.error("Please provide both URLs.")

    df_a_raw = st.session_state['data_store']['Arena']['df_a']
    df_b_raw = st.session_state['data_store']['Arena']['df_b']

    if df_a_raw is not None and df_b_raw is not None:
        st.divider()
        st.markdown("### Step 2: Select Data to Analyze")
        st.caption("Select specific comments to compare using the checkboxes. If none selected, the top N rows will be analyzed.")
        
        c_edit_a, c_edit_b = st.columns(2)
        with c_edit_a:
            st.markdown(f"**Contender A** ({len(df_a_raw)} items)")
            edited_a = st.data_editor(
                df_a_raw,
                column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                disabled=["content", "agent_id", "timestamp", "is_bot", "likes"],
                key="editor_arena_a", height=300, hide_index=True
            )
        with c_edit_b:
            st.markdown(f"**Contender B** ({len(df_b_raw)} items)")
            edited_b = st.data_editor(
                df_b_raw,
                column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                disabled=["content", "agent_id", "timestamp", "is_bot", "likes"],
                key="editor_arena_b", height=300, hide_index=True
            )

        st.divider()
        c_action, c_limit = st.columns([1, 1])
        with c_limit:
            max_analyze = st.number_input("Max Rows to Analyze (if no manual selection)", 5, 100, 20)
        with c_action:
            st.write("") 
            st.write("") 
            start_analysis = st.button("Step 3: Run Comparative Analysis", type="primary", disabled=not key)

        if start_analysis:
            subset_a = edited_a[edited_a.Select]
            if subset_a.empty: subset_a = edited_a.head(max_analyze)
            
            subset_b = edited_b[edited_b.Select]
            if subset_b.empty: subset_b = edited_b.head(max_analyze)

            prog_a = st.progress(0)
            res_a = []
            st.markdown("**Analyzing Contender A...**")
            for i, (_, row) in enumerate(subset_a.iterrows()):
                res_a.append(analyze_fallacies(row['content'], api_key=key))
                prog_a.progress((i + 1) / len(subset_a))
            final_a = pd.concat([subset_a.reset_index(drop=True), pd.DataFrame(res_a)], axis=1)
            st.session_state['data_store']['Arena']['analyzed_a'] = final_a

            prog_b = st.progress(0)
            res_b = []
            st.markdown("**Analyzing Contender B...**")
            for i, (_, row) in enumerate(subset_b.iterrows()):
                res_b.append(analyze_fallacies(row['content'], api_key=key))
                prog_b.progress((i + 1) / len(subset_b))
            final_b = pd.concat([subset_b.reset_index(drop=True), pd.DataFrame(res_b)], axis=1)
            st.session_state['data_store']['Arena']['analyzed_b'] = final_b
            
            st.rerun()

    res_df_a = st.session_state['data_store']['Arena']['analyzed_a']
    res_df_b = st.session_state['data_store']['Arena']['analyzed_b']

    if res_df_a is not None and res_df_b is not None:
        st.divider()
        st.header("Match Results")
        
        c1, c2, c3 = st.columns(3)
        agg_a = res_df_a['aggression'].mean()
        agg_b = res_df_b['aggression'].mean()
        delta_agg = agg_a - agg_b
        bots_a = len(res_df_a[res_df_a['is_bot']==True])
        bots_b = len(res_df_b[res_df_b['is_bot']==True])
        delta_bots = bots_a - bots_b
        fallacy_a = len(res_df_a[res_df_a['has_fallacy']==True])
        fallacy_b = len(res_df_b[res_df_b['has_fallacy']==True])
        
        c1.metric("Avg Aggression (A vs B)", f"{agg_a:.1f} vs {agg_b:.1f}", f"{delta_agg:.1f}")
        c2.metric("Bot Count (A vs B)", f"{bots_a} vs {bots_b}", f"{delta_bots}")
        c3.metric("Fallacies (A vs B)", f"{fallacy_a} vs {fallacy_b}")
        
        st.subheader("Visual Comparison")
        res_df_a['Source'] = 'Contender A'
        res_df_b['Source'] = 'Contender B'
        combined = pd.concat([res_df_a, res_df_b])
        
        chart = alt.Chart(combined).mark_bar().encode(
            x=alt.X('Source', title=None),
            y=alt.Y('mean(aggression)', title='Avg Aggression'),
            color=alt.Color('Source', scale=alt.Scale(domain=['Contender A', 'Contender B'], range=['#3b82f6', '#f97316'])),
            tooltip=['Source', 'mean(aggression)']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        
        with st.expander("Detailed Comparison Data"):
            st.dataframe(combined[['Source', 'content', 'aggression', 'fallacy_type', 'is_bot']])

# ==========================================
# MODULE 5: LIVE RADAR (RSS/REDDIT)
# ==========================================
elif mode == "5. Live Radar (RSS/Reddit)":
    st.header("5. Live Radar (Crisis Alert System)")
    st.caption("Monitor live RSS feeds or subreddits to intercept escalating disinformation and aggression in real-time.")
    
    if "GEMINI_API_KEY" in st.secrets: key = st.secrets["GEMINI_API_KEY"]
    else: key = st.text_input("API Key", type="password")
    
    c_radar1, c_radar2 = st.columns([2, 1])
    with c_radar1:
        feed_url = st.text_input("Enter RSS Feed, Subreddit, or News Keyword", placeholder="E.g., ansa, bbc, repubblica, reddit.com/r/worldnews")
    with c_radar2:
        max_entries = st.number_input("Entries to Fetch", 5, 50, 15)
        fetch_btn = st.button("Step 1: Fetch Live Feed", type="primary", use_container_width=True)

    # --- STEP 1: FETCH DATA ---
    if fetch_btn and feed_url:
        with st.spinner("Intercepting live feed..."):
            try:
                user_input = feed_url.strip().lower()
                news_shortcuts = {
                    "ansa": "https://www.ansa.it/sito/ansait_rss.xml",
                    "repubblica": "https://www.repubblica.it/rss/homepage/rss2.0.xml",
                    "corriere": "http://xml2.corriereobjects.it/rss/homepage.xml",
                    "bbc": "http://feeds.bbci.co.uk/news/world/rss.xml",
                    "cnn": "http://rss.cnn.com/rss/edition.rss",
                    "nytimes": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"
                }
                
                actual_url = feed_url
                if user_input in news_shortcuts:
                    actual_url = news_shortcuts[user_input]
                elif "reddit.com/r/" in user_input and not user_input.endswith(".rss"):
                    if actual_url.endswith("/"): actual_url = actual_url[:-1]
                    actual_url += "/new/.rss"
                
                feed = feedparser.parse(actual_url)
                
                if not feed.entries:
                    st.error("Could not fetch entries. Check the URL or formatting.")
                else:
                    entries_data = []
                    for entry in feed.entries[:int(max_entries)]:
                        clean_summary = re.sub(r'<[^>]+>', '', entry.get('summary', ''))
                        text_to_analyze = f"TITLE: {entry.get('title', '')}\nCONTENT: {clean_summary[:500]}"
                        entries_data.append({
                            'Select': False,
                            'timestamp': entry.get('published', 'Unknown'),
                            'content': text_to_analyze,
                            'link': entry.get('link', '')
                        })
                    
                    st.session_state['data_store']['Radar']['df'] = pd.DataFrame(entries_data)
                    st.session_state['data_store']['Radar']['analyzed'] = None
                    st.success(f"✅ Intercepted {len(entries_data)} items from {feed.feed.get('title', 'Feed')}.")
            except Exception as e:
                st.error(f"Radar Fetch Error: {str(e)}")

    # --- STEP 2: SELECTION AND ANALYSIS ---
    radar_df = st.session_state['data_store'].get('Radar', {}).get('df')
    
    if radar_df is not None:
        st.divider()
        st.markdown("### Step 2: Select News to Analyze")
        st.caption("Select specific items using checkboxes. If none selected, the top N rows will be analyzed.")
        
        edited_radar = st.data_editor(
            radar_df,
            column_config={"Select": st.column_config.CheckboxColumn(required=True)},
            disabled=["timestamp", "content", "link"],
            key="editor_radar",
            height=300,
            hide_index=True,
            use_container_width=True
        )
        
        c_action, c_limit = st.columns([1, 1])
        with c_limit:
            max_analyze = st.number_input("Max Rows to Analyze (if no manual selection)", 1, len(radar_df), min(10, len(radar_df)))
        with c_action:
            st.write("") 
            st.write("") 
            
            selected = edited_radar['Select'].sum()
            
            if selected > 0:
                btn_testo = f"Step 3: Run Threat Analysis ({selected})"
            else:
                btn_testo = f"Step 3: Run Threat Analysis (Batch Top {max_analyze})"
                
            analyze_btn = st.button(btn_testo, type="primary", disabled=not key)
        
        if analyze_btn:
            subset = edited_radar[edited_radar.Select]
            if subset.empty:
                subset = edited_radar.head(max_analyze)
                st.info(f"Using top {len(subset)} rows (Auto-Batch).")
            else:
                st.success(f"Using {len(subset)} manually selected rows.")
                
            prog = st.progress(0)
            res = []
            st.markdown("**Running Threat Analysis...**")
            for i, (_, row) in enumerate(subset.iterrows()):
                res.append(analyze_fallacies(row['content'], api_key=key, persona="Crisis & Sentiment Analyst"))
                prog.progress((i + 1) / len(subset))
            
            final_df = pd.concat([subset.reset_index(drop=True), pd.DataFrame(res)], axis=1)
            st.session_state['data_store']['Radar']['analyzed'] = final_df
            st.rerun()
    
    # --- STEP 3: DISPLAY RESULTS ---
    analyzed_radar = st.session_state['data_store'].get('Radar', {}).get('analyzed')
    if analyzed_radar is not None:
        st.divider()
        avg_aggression = analyzed_radar['aggression'].mean()
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Live Aggression Index", f"{avg_aggression:.1f}/10")
        col_m2.metric("Flagged Issues", len(analyzed_radar[analyzed_radar['has_fallacy']==True]))
        col_m3.metric("Monitored Items", len(analyzed_radar))
        
        if avg_aggression > 6:
            st.error("🚨 **CRISIS ALERT:** The current feed is exhibiting highly aggressive or toxic sentiment.")
        elif avg_aggression > 4:
            st.warning("⚠️ **ELEVATED TENSION:** The feed contains moderate aggression and polarization.")
        else:
            st.success("✅ **STABLE:** The feed is relatively neutral and calm.")
        
        st.subheader("Live Feed Feedbacks")
        for _, r in analyzed_radar.iterrows():
            with st.container(border=True):
                st.caption(f"🕒 {r['timestamp']} | 🔗 [Source Link]({r['link']}) | **Agg:** {r['aggression']}/10")
                st.write(r['content'][:300] + "...")
                if r['has_fallacy']:
                    st.error(f"🛑 **{r['fallacy_type']}**: {r['explanation']}")
                else:
                    st.success(f"✅ {r.get('explanation', 'Clear.')}")
        st.divider()
        st.subheader("📥 Export Radar Intel")
        st.caption("Download the monitored items for your intelligence reports.")
        
        # Puliamo le colonne per l'esportazione
        export_df = analyzed_radar.drop(columns=['Select'], errors='ignore')
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Radar Alert Report (CSV)",
            data=csv_data,
            file_name=f"RAP_Radar_Alert_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            type="primary"
        )

# ==========================================
# MODULE 6: DEEP DOCUMENT ORACLE (RAG)
# ==========================================
elif mode == "6. Deep Document Oracle (RAG)":
    st.header("6. Deep Document Oracle")
    st.caption("Upload massive PDFs (e.g., manifestos, contracts, books) and find contradictions and extract deep facts without traditional RAG limits.")

    if "GEMINI_API_KEY" in st.secrets: key = st.secrets["GEMINI_API_KEY"]
    else: key = st.text_input("API Key", type="password")

    uploaded_files = st.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if 'doc_full_text' not in st.session_state or st.button("Process Documents"):
            with st.spinner("Extracting text from all documents..."):
                full_text = ""
                for f in uploaded_files:
                    txt = extract_text_from_pdf(f)
                    if txt:
                        full_text += f"\n\n--- DOCUMENT: {f.name} ---\n\n{txt}"
                
                st.session_state['doc_full_text'] = full_text
                st.success(f"Processed {len(full_text)} characters across {len(uploaded_files)} documents. The Oracle is ready.")
        
        if 'doc_full_text' in st.session_state and st.session_state['doc_full_text']:
            st.divider()
            
            # --- MODIFICA KNOWLEDGE GRAPH (Modulo 6) ---
            with st.expander("Extract Document Power Network (Knowledge Graph)", expanded=False):
                st.caption("Automatically scan the document to map relationships between People, Organizations, and Locations.")
                if st.button("Generate Power Graph"):
                    with st.spinner("Extracting entities and relationships (this may take a minute)..."):
                        graph_prompt = f"""
                        Analyze this document and extract the top 12 most important relationships between entities (People, Organizations, Locations).
                        CRITICAL RULES:
                        1. The "relation" field MUST BE ULTRA-SHORT (maximum 1 to 3 words, e.g., "CEO", "Owned by", "Funded", "Opposes").
                        2. Keep entity names short and clean.
                        Return ONLY a valid JSON format like this:
                        {{ "relations": [ {{"source": "Entity 1", "target": "Entity 2", "relation": "Short Label"}} ] }}
                        
                        DOCUMENT:
                        {st.session_state['doc_full_text'][:30000]}
                        """
                        try:
                            client = genai.Client(api_key=key)
                            graph_res = client.models.generate_content(model='gemini-2.0-flash', contents=graph_prompt)
                            graph_json = extract_json(graph_res.text)
                            
                            fig_doc = plot_document_entity_graph(graph_json)
                            if fig_doc:
                                st.pyplot(fig_doc)
                            else:
                                st.error("Not enough clear relationships found to build a graph.")
                        except Exception as e:
                            st.error(f"Graph Extraction Error: {str(e)}")
            st.divider()
            # ---------------------------------------------
            
            st.subheader("Chat with the Oracle")
            
            for message in st.session_state.doc_oracle_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            if prompt := st.chat_input("Ask the Deep Oracle (e.g., 'Find all contradictions in chapter 3')..."):
                st.session_state.doc_oracle_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Scouring massive document context..."):
                        response = ask_document_oracle(st.session_state['doc_full_text'], prompt, key)
                        st.markdown(response)
                        st.session_state.doc_oracle_history.append({"role": "assistant", "content": response})
