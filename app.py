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
import hashlib
import plotly.graph_objects as go
import plotly.express as px
import urllib.request
import cv2
import math
from PIL import ImageDraw, ImageChops, ImageEnhance
import urllib.parse
from cv2 import GaussianBlur
import requests

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
if 'global_entities' not in st.session_state: st.session_state['global_entities'] = set()
if 'seen_radar_links' not in st.session_state: st.session_state['seen_radar_links'] = set()

if 'data_store' not in st.session_state:
    st.session_state['data_store'] = {
        'CSV File Upload': {'df': None, 'analyzed': None, 'summary': None},
        'YouTube Link': {'df': None, 'analyzed': None, 'summary': None},
        'Raw Text Paste': {'df': None, 'analyzed': None, 'summary': None},
        'Telegram Dump (JSON)': {'df': None, 'analyzed': None, 'summary': None},
        'Reddit Native (OSINT)': {'df': None, 'analyzed': None, 'summary': None},
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

# --- HELPER: ERROR LEVEL ANALYSIS (ELA) ---
def perform_ela(image, quality=90):
    """Calculates the Error Level Analysis of an image to detect Photoshop manipulations."""
    try:
        # Ensure the image is in RGB mode
        original = image.convert('RGB')
        
        # Temporarily resave the image at a lower quality
        buffer = io.BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)
        
        # Calculate the absolute difference between the original and the compressed image
        ela_image = ImageChops.difference(original, resaved)
        
        # Visually amplify the difference to make it visible to the human eye
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0: max_diff = 1
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        return ela_image
    except Exception as e:
        return None

# --- HELPER: OPSEC FACE ANONYMIZATION (ULTIMATE BERSERKER CALIBRATION) ---
def anonymize_faces(image):
    """
    Executes a dual sweep with maximum exhaustive search (scaleFactor=1.05).
    Uses a highly precise NMS (Non-Maximum Suppression) style logic to filter overlaps
    without deleting faces of people standing close to each other.
    """
    try:
        # 1. Convert PIL Image to OpenCV format
        img_cv = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 2. Load BOTH models
        frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # 3. SWEEP 1: Frontal Scan (Aggressive search 1.05, but minNeighbors=4 blocks background noise)
        frontal_faces = frontal_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20)
        )
        
        # 4. SWEEP 2: Profile Scan
        profile_faces = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20)
        )
        
        # Convert tuples to lists
        f_faces = list(frontal_faces) if len(frontal_faces) > 0 else []
        p_faces = list(profile_faces) if len(profile_faces) > 0 else []
        all_faces = f_faces + p_faces
        
        # 5. SURGICAL ANTI-OVERLAP PROTOCOL
        final_faces = []
        for (x, y, w, h) in all_faces:
            overlap = False
            for (fx, fy, fw, fh) in final_faces:
                cx, cy = x + w/2, y + h/2
                fcx, fcy = fx + fw/2, fy + fh/2
                dist = ((cx - fcx)**2 + (cy - fcy)**2)**0.5
                
                # ONLY discard if the centers are EXTREMELY close (less than half the face width)
                # This prevents deleting two different people standing shoulder-to-shoulder
                if dist < (min(w, fw) * 0.4):
                    overlap = True
                    break
            
            if not overlap:
                final_faces.append((x, y, w, h))
        
        # 6. Apply Dynamic Heavy Blur
        for (x, y, w, h) in final_faces:
            face_roi = img_cv[y:y+h, x:x+w]
            
            # Calculate blur intensity
            k_size = (w // 2) | 1 
            if k_size < 3: k_size = 3
            
            # Sigma=50 guarantees a very dense, completely opaque blur
            blurred_face = cv2.GaussianBlur(face_roi, (k_size, k_size), 50)
            img_cv[y:y+h, x:x+w] = blurred_face
            
        # 7. Return the final image
        img_final = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_final), len(final_faces)
        
    except Exception as e:
        return image, 0
    
# --- HELPER: EXIF GPS SNIPER ---
def get_decimal_from_dms(dms, ref):
    """Converts Degrees, Minutes, Seconds to Decimal Coordinates."""
    degrees, minutes, seconds = dms[0], dms[1], dms[2]
    dec = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if ref in ['S', 'W']: dec = -dec
    return dec

def get_exif_location(image):
    """Extracts hidden GPS coordinates from an image's EXIF data."""
    try:
        exif = image._getexif()
        if not exif: return None
        geotagging = {}
        for (idx, tag) in TAGS.items():
            if tag == 'GPSInfo' and idx in exif:
                for (key, val) in GPSTAGS.items():
                    if key in exif[idx]:
                        geotagging[val] = exif[idx][key]
        
        if 'GPSLatitude' in geotagging and 'GPSLongitude' in geotagging:
            lat = get_decimal_from_dms(geotagging['GPSLatitude'], geotagging.get('GPSLatitudeRef', 'N'))
            lon = get_decimal_from_dms(geotagging['GPSLongitude'], geotagging.get('GPSLongitudeRef', 'E'))
            return lat, lon
    except:
        pass
    return None

# --- HELPER: DIGITAL FORENSICS HASHING ---
def calculate_sha256(file_bytes):
    """Generates a cryptographic hash for Chain of Custody proof."""
    if not file_bytes: return "N/A"
    return hashlib.sha256(file_bytes).hexdigest()

# --- HELPER: FORENSIC STORYBOARD GENERATOR ---
def create_video_storyboard(video_bytes, num_frames=12):
    """Dismembers the video into exactly 12 frames and merges them into a single grid image."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    
    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0: return None
        
    interval = max(1, total_frames // num_frames)
    frames = []
    
    for i in range(num_frames):
        frame_id = i * interval
        if frame_id >= total_frames: frame_id = total_frames - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            # Resize to prevent memory explosion
            pil_img.thumbnail((400, 400)) 
            timestamp = frame_id / fps if fps > 0 else 0
            frames.append((pil_img, timestamp))
            
    cap.release()
    os.unlink(tmp_path)
    
    if not frames: return None
        
    # Create the grid (3 columns)
    cols = 3
    rows = math.ceil(len(frames) / cols)
    w, h = frames[0][0].size
    padding_y = 40
    
    grid_img = Image.new('RGB', (cols * w, rows * (h + padding_y)), color='black')
    draw = ImageDraw.Draw(grid_img)
    
    for i, (img, ts) in enumerate(frames):
        row = i // cols
        col = i % cols
        x = col * w
        y = row * (h + padding_y)
        grid_img.paste(img, (x, y + padding_y))
        # Print the exact timestamp on the frame
        draw.text((x + 10, y + 10), f"Frame Time: {ts:.2f}s", fill="red")
        
    # Return the PIL image object directly instead of raw bytes
    return grid_img

# --- HELPER: AUDIO WAVEFORM GENERATION ---
def generate_audio_waveform(audio_bytes):
    """Generates a visual waveform from audio bytes for forensic inspection."""
    try:
        # Convert bytes to a format numpy can read (int16)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(audio_data, color='#ef4444', linewidth=0.5)
        ax.axis('off')
        ax.set_title("Forensic Audio Waveform (Raw Signal)", color='white', size=10)
        fig.patch.set_alpha(0) # Transparent background
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return buf.getvalue()
    except:
        return None

# --- HELPER: PII SANITIZER (OPERATION BLACKOUT - CIA STYLE) ---
def sanitize_pii(text):
    if not isinstance(text, str): return text
    # Replace sensitive data with solid black blocks (Classified Redaction)
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '██████████', text)
    text = re.sub(r'\b\+?\d{1,3}?[-.\s]?\(?\d{2,3}?\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '██████████', text)
    text = re.sub(r'\b[A-Z]{2}\d{2}[a-zA-Z0-9]{11,30}\b', '████████████████', text)
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '████████', text)
    return text

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
    - CRITICAL RULE: Always cite the exact page number(s) to back up your claims, using the [PAGE X] tags provided in the text format (e.g., "According to the contract [PAGE 12]...").
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

# --- HELPER: PDF EXTRACTOR & SCRAPERS ---
@st.cache_data
def extract_text_from_pdf(file_obj):
    try:
        pdf_reader = pypdf.PdfReader(file_obj)
        text = ""
        # Enumerate pages starting from 1 to track page numbers
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            content = page.extract_text()
            if content: 
                # Inject page markers directly into the text stream
                text += f"\n\n[PAGE {page_num}]\n{content}"
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

# --- NATIVE REDDIT OSINT SCRAPER ---
@st.cache_data(show_spinner=False)
def scrape_reddit_native(subreddit, limit=50):
    try:
        # Clean the input (allow users to type "r/worldnews" or just "worldnews")
        sub = subreddit.replace('r/', '').replace('https://www.reddit.com/r/', '').strip('/')
        url = f"https://www.reddit.com/r/{sub}/new.json?limit={limit}"
        
        # Mask the request as a normal web browser to avoid Reddit's API blocks
        req = urllib.request.Request(
            url, 
            data=None, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 RAP_OSINT_Bot/1.0'}
        )
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        
        posts = []
        for child in data.get('data', {}).get('children', []):
            post = child.get('data', {})
            content = post.get('selftext', '')
            title = post.get('title', '')
            full_text = f"{title}\n{content}".strip()
            
            if full_text:
                posts.append({
                    'agent_id': post.get('author', 'Unknown'),
                    'timestamp': datetime.fromtimestamp(post.get('created_utc', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'content': full_text,
                    'likes': post.get('score', 0)
                })
        if not posts: return None
        return pd.DataFrame(posts)
    except Exception as e:
        return None

# --- HELPER: RAW PASTE PARSER ---
def parse_raw_paste(raw_text):
    """
    Parses pasted text. Cuts strictly at the FIRST comma if the header is present,
    preventing commas inside the comments from breaking the data structure.
    """
    lines = [line.strip() for line in raw_text.strip().split('\n') if line.strip()]
    data = []
    now = datetime.now()
    
    has_header = False
    if lines and lines[0].lower().replace(" ", "") == "agent_id,content":
        has_header = True
        lines = lines[1:]
        
    for i, line in enumerate(lines):
        if has_header and ',' in line:
            parts = line.split(',', 1)
            agent = parts[0].strip()
            content = parts[1].strip()
        else:
            agent = "Paste_Source"
            content = line
            
        data.append({
            "agent_id": agent,
            "timestamp": (now - timedelta(minutes=i*10)).strftime("%Y-%m-%d %H:%M:%S"),
            "content": content,
            "likes": 0
        })
        
    return pd.DataFrame(data)

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

# --- HYBRID ANALYZER (WITH AUTO-TRANSLATE) ---
@st.cache_data(show_spinner=False)
def analyze_fallacies_cached(text, api_key, context_info="", persona="Logic & Fact Analysis Engine", target_lang="English"):
    if not text or len(str(text)) < 3: return sanitize_response(None)
    for attempt in range(3):
        try:
            client = genai.Client(api_key=api_key)
            is_long = len(text) > 2000
            common_instructions = f"""
            You are a {persona}. CONTEXT: "{context_info}"
            CRITICAL MULTILINGUAL RULE: The input text might be in any language. 
            You MUST conduct the analysis and write ALL your JSON responses exclusively in {target_lang}.
            TASKS:
            - explanation: Brief Analysis in {target_lang}.
            - main_topic: Central theme in {target_lang}.
            - micro_topic: 1-2 keywords in {target_lang}.
            - target: Target Entity.
            - primary_emotion: [Anger, Fear, Disgust, Sadness, Joy, Surprise, Neutral] in {target_lang}.
            - archetype: [Instigator, Loyalist, Troll, Rational Skeptic, Observer] in {target_lang}.
            - propaganda_tactic: Identify Military Info-War Tactics (e.g., Firehose of Falsehood, Astroturfing, Dead Cat, Whataboutism) if present.
            RULES: Opinions/Sarcasm -> 'has_fallacy': false. You MUST still find standard logical fallacies in 'fallacy_type'.
            RESPONSE (Strict JSON):
            {{ "has_fallacy": bool, "fallacy_type": "Name", "propaganda_tactic": "Name", "explanation": "Text", "correction": "Text", "main_topic": "Text", "micro_topic": "Text", "target": "Text", "primary_emotion": "Text", "archetype": "Text", "relevance": "Text", "counter_reply": "Text", "sentiment": "Text", "translated_text": "Translate the original text into {target_lang} here", "aggression": 0, "sophistication": 0, "hostile_intent": 0 }}
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

def analyze_fallacies(text, api_key=None, context_info="", persona="Logic & Fact Analysis Engine", target_lang="English"):
    if not api_key: return {"has_fallacy": True}
    res = analyze_fallacies_cached(text, api_key, context_info, persona, target_lang)
    increment_counter(len(text), 500)
    return res

# --- HELPER: COGNITIVE EDITOR (MULTIMODAL + AI DETECTOR) ---
def cognitive_rewrite(text, api_key, media_data=None, media_type="image", target_lang="English"):
    if (not text or len(str(text)) < 3) and not media_data: return None
    try:
        client = genai.Client(api_key=api_key)
        
        prompt_text = f"""
        You are a highly paranoid Strategic Intelligence Investigator and a Military-Grade Digital Forensics Expert. You NEVER trust the narrative of a video or image. You ONLY trust physics, pixels, and temporal consistency.
        
        YOUR TASKS:
        1. AI FORENSICS (THE PARANOIA PROTOCOL): Analyze media for AI GENERATION (Sora, Runway, Midjourney, etc.). Score 0-20 (Natural), 40-70 (Manipulated), 70-100 (Fully AI Generated).
           CRITICAL INSTRUCTION: DO NOT get distracted by what is happening in the scene. Focus strictly on HOW the matter behaves. Look specifically for:
           - TEMPORAL MORPHING: Do objects, limbs, or faces change volume, shape, or structure from one second to another?
           - MATTER COMPENETRATION (CLIPPING): Do solid objects pass through each other? (e.g., hands melting into clothing, feet sinking into solid concrete, a person merging with an animal).
           - KINEMATICS & GRAVITY: Do movements look "floaty", lack real momentum, or defy gravity? Does the center of mass shift impossibly?
           - BACKGROUND WARPING: Does the background or static environment stretch, drag, or melt when a foreground subject moves past it?
           - TEXTURE HALLUCINATION: Does text, fur, or fabric boil, shift, or turn into illegible alien symbols as the camera moves?
           If you detect EVEN ONE of these physics-breaking anomalies, you MUST set the 'ai_generated_probability' score above 85% and detail the exact hallucination.
           
        2. IDENTITY VERIFICATION: Identify any famous people. Evaluate if their appearance and movements are physically consistent.
        3. ADVANCED GEO-INT (Shadow Geolocation): 
           - IF THE MEDIA IS AI-GENERATED (Score > 60): Output exactly "Fictional AI-Generated Environment - Geolocation not applicable."
           - IF THE MEDIA IS REAL: Deduce geographic location by identifying micro-clues (street signs, architectural styles, car plates).
        4. SYLLOGISM MACHINE: Deconstruct the core argument of the text/speech into formal logical steps.
        5. VIDEO TIMELINE: MANDATORY IF YOU SEE A STORYBOARD GRID OR A VIDEO: Provide at least 5-8 timestamp objects in the 'video_timeline' array. Read the red timestamp text on the frames (e.g., "0.00s", "2.50s") and use those for the 'timestamp' field. Detail the EXACT physical anomalies or morphing glitches at each frame. Do NOT describe the plot; describe the physical artifacts.
        6. AGGRESSION: Score emotional intensity from 0 to 10.
        7. GHOST READER (OCR): Extract ANY visible text (signs, documents, screens) from the media. You MUST provide BOTH the original text AND its direct translation in the 'ocr_extraction' field. Format it elegantly with a double line break between them, EXACTLY like this:
        "**Original:** [text]
        
        **Translation ({target_lang}):** [text]"
        
        CRITICAL LANGUAGE RULE: 
        You MUST write EVERY single output field (explanation, ai_analysis, shadow_geolocation, etc.) and translations STRICTLY in {target_lang}. Ignore the original language of the media or text input for your output language.
        
        JSON OUTPUT RULES (Keep keys in English):
        - "fallacy_type": Name of the issue/fallacy.
        - "explanation": Comprehensive analysis.
        - "ai_analysis": Detailed forensic breakdown of physics violations and morphing.
        - "syllogism_breakdown": Array of objects with 'step', 'text', 'flaw'.
        - "video_timeline": Array of objects with 'timestamp', 'ai_score', 'details' (focus on glitches, not plot).
        - "shadow_geolocation": String with detailed geographic deduction.
        - "aggression": Integer (0 to 10).
        - "transcript": EXACT word-for-word transcription. Leave empty if none.
        - "ocr_extraction": String with all visible text extracted from the image/video.
        
        RESPONSE (Strict JSON):
        {{
            "has_fallacy": true/false,
            "fallacy_type": "",
            "explanation": "",
            "rewritten_text": "",
            "transcript": "",
            "facts": [],
            "aggression": 0,
            "ai_generated_probability": 0,
            "ai_analysis": "",
            "voice_stress_score": 0,
            "shadow_geolocation": "",
            "syllogism_breakdown": [],
            "video_timeline": [],
            "search_sources": [],
            "ocr_extraction": ""
        }}
        """
        
        contents = [prompt_text]
        if text: contents.append(f"Input Text/Context: {text[:10000]}")
        
        if media_data is not None:
            if media_type == "image":
                # Ensure compatibility whether it's raw bytes or a PIL Image
                if isinstance(media_data, bytes):
                    contents.append(types.Part.from_bytes(data=media_data, mime_type="image/jpeg"))
                else:
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

# --- INTERACTIVE NETWORK GRAPH (PLOTLY) ---
def plot_network_graph(df):
    if 'target' not in df.columns or 'micro_topic' not in df.columns: return None
    valid = df[(df['target'] != "Unknown") & (df['target'] != "None") & (df['micro_topic'] != "Unknown")]
    if valid.empty: return None
    
    G = nx.Graph()
    for _, row in valid.iterrows():
        target = f"{row['target']}"
        topic = f"{row['micro_topic']}"
        G.add_edge(target, topic, weight=1)
        
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#475569'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='#ef4444',
            size=15,
            line_width=2,
            line_color='white'))
            
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(text='Interactive Entity-Narrative Network', font=dict(size=16)),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
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
    "3. Cognitive Editor (Text/Image/Audio/Video)", 
    "4. Comparison Test (A/B Testing)",
    "5. Live Radar (RSS/Reddit)",
    "6. Deep Document Oracle (RAG)"
])

# --- GLOBAL LANGUAGE SELECTOR ---
st.sidebar.markdown("---")
world_languages = [
    "English", "Italiano", "Español", "Français", "Deutsch", "Português",
    "Русский (Russian)", "العربية (Arabic)", "中文 (Chinese)", 
    "日本語 (Japanese)", "فارسی (Persian)", "हिन्दी (Hindi)", 
    "한국어 (Korean)", "Türkçe (Turkish)"
]
report_language = st.sidebar.selectbox("Global Output Language", world_languages, index=0)

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

# --- CLEAR WORKSPACE BUTTON ---
st.sidebar.markdown("---")
if st.sidebar.button("Clear Workspace", type="primary", help="Erase all current session data and start a clean investigation"):
    # Delete specific keys to reset the dashboard state
    keys_to_clear = ['data_store', 'oracle_history', 'doc_oracle_history', 'doc_full_text', 'global_entities']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    # Force a page reload to apply changes
    st.rerun()

# --- OSINT ARSENAL (SIDEBAR) ---
st.sidebar.markdown("---")
with st.sidebar.expander("OSINT Arsenal (Target Intel)", expanded=False):
    st.markdown("Generate tactical search vectors for any target entity.")
    osint_target = st.text_input("Enter Target Entity (Name/Company/IP):", placeholder="e.g. John Doe, CyberCorp")
    
    if osint_target:
        safe_t = urllib.parse.quote(osint_target)
        
        st.markdown("**Google Dorks (Deep Web)**")
        st.markdown(f"- [Find Leaked PDF/DOCX](https://www.google.com/search?q=ext:pdf+OR+ext:docx+%22{safe_t}%22+%22confidential%22+OR+%22internal%22)")
        st.markdown(f"- [Find Open Directories](https://www.google.com/search?q=intitle:%22index+of%22+%22{safe_t}%22)")
        st.markdown(f"- [Find Pastebin Leaks](https://www.google.com/search?q=site:pastebin.com+%22{safe_t}%22)")
        st.markdown(f"- [LinkedIn Employee Search](https://www.google.com/search?q=site:linkedin.com/in+%22{safe_t}%22)")
        
        st.markdown("**Infrastructure & Ports**")
        st.markdown(f"- [Shodan (IoT & Servers)](https://www.shodan.io/search?query={safe_t})")
        st.markdown(f"- [Censys (Certificates)](https://search.censys.io/search?resource=hosts&q={safe_t})")
        
        st.markdown("**Social Intelligence**")
        st.markdown(f"- [Twitter/X Advanced Search](https://twitter.com/search?q=%22{safe_t}%22&src=typed_query)")
        st.markdown(f"- [Reddit Mention Tracker](https://www.reddit.com/search/?q=%22{safe_t}%22)")

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

    with st.expander("View Neural Infection Network", expanded=False):
        st.caption("Topological visualization of infection clusters.")
        sample_size = min(150, n_agents)
        
        if topology == "Echo Chambers (Clusters)": G = nx.caveman_graph(5, sample_size // 5)
        elif topology == "Influencer Network (Hubs)": G = nx.barabasi_albert_graph(sample_size, 2, seed=42)
        else: G = nx.erdos_renyi_graph(sample_size, 0.05, seed=42)
        
        node_colors = []
        for i in range(len(G.nodes())):
            if current[i] > 0.8: node_colors.append('#ef4444') # Infected (Red)
            elif current[i] > 0.3: node_colors.append('#f97316') # At risk (Orange)
            else: node_colors.append('#3b82f6') # Healthy (Blue)
            
        fig_net, ax_net = plt.subplots(figsize=(10, 5))
        pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)
        nx.draw(G, pos, node_color=node_colors, edge_color='#e0e0e0', node_size=100, alpha=0.8, ax=ax_net)
        ax_net.set_title(f"Network Topology State (Day {steps})")
        st.pyplot(fig_net)
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

    # --- PDF EXPORT FOR WARGAME ---
    st.markdown("---")
    st.subheader("Export Tactical Report")
    st.caption("Generate a summary document with the simulation parameters and results.")
    
    if st.button("Generate Wargame PDF", type="primary"):
        with st.spinner("Compiling the report..."):
            pdf = PDFReport()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Wargame specific header
            pdf.set_font("Helvetica", 'B', 14)
            pdf.cell(0, 10, "Simulation Dossier: Information Warfare", 0, 1)
            pdf.ln(5)
            
            # Section 1: Parameters
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(0, 10, "Base Scenario Parameters:", 0, 1)
            pdf.set_font("Helvetica", '', 11)
            pdf.cell(0, 8, f"- Network Topology: {topology}", 0, 1)
            pdf.cell(0, 8, f"- Population Size: {n_agents} nodes", 0, 1)
            pdf.cell(0, 8, f"- Initial Infection Rate (Patient Zero): {bot_pct*100:.1f}%", 0, 1)
            pdf.cell(0, 8, f"- Blue Team Countermeasure: {defense}", 0, 1)
            pdf.cell(0, 8, f"- Time Horizon: {steps} cycles (days)", 0, 1)
            pdf.ln(5)
            
            # Section 2: Results
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(0, 10, "Operation Results & Impact:", 0, 1)
            pdf.set_font("Helvetica", '', 11)
            pdf.cell(0, 8, f"- Final Infection Rate: {final_rate:.1f}%", 0, 1)
            pdf.cell(0, 8, f"- Net Variation (Delta): {delta:.1f}%", 0, 1)
            pdf.ln(5)
            
            # Section 3: Automated Assessment
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(0, 10, "Strategic Assessment:", 0, 1)
            pdf.set_font("Helvetica", 'I', 11)
            
            if delta > 5:
                esito = "CRITICAL: The adopted countermeasure proved ineffective. The cognitive infection spread aggressively, bypassing network defenses."
            elif delta < -5:
                esito = "SUCCESS: The countermeasure had an excellent containment impact, drastically reducing the presence of the informational pathogen."
            else:
                esito = "STABLE: The situation remained mostly unchanged. Defenses held the initial shockwave but failed to eradicate the threat."
                
            # Clean encoding for FPDF
            pdf.multi_cell(0, 6, esito.encode('latin-1', 'replace').decode('latin-1'))
            
            pdf_bytes = bytes(pdf.output())
            
        st.download_button(
            label="Download Wargame Report (PDF)", 
            data=pdf_bytes, 
            file_name=f"RAP_Wargame_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", 
            mime="application/pdf", 
            type="primary"
        )

# ==========================================
# MODULE 2: SOCIAL DATA ANALYSIS
# ==========================================
elif mode == "2. Social Data Analysis (Universal)":
    st.header("2. Social Data Analysis")
    st.sidebar.header("Settings")
    st.sidebar.header("Settings")
    
    # --- CUSTOM AI LENS (PERSONA) ---
    preset_personas = [
        "Strategic Intelligence Analyst", 
        "Mass Psychologist (Emotional)", 
        "Legal Consultant (Defamation/Risk)", 
        "Campaign Manager (Opportunity)",
        "Custom (Define your own role...)"
    ]
    
    selected_persona = st.sidebar.selectbox("Analysis Lens (Persona)", preset_personas)
    
    if selected_persona == "Custom (Define your own role...)":
        # Allow the user to input any specific professional perspective
        persona = st.sidebar.text_input("Enter Custom Persona", value="Cybersecurity Expert hunting for coordination", help="Define the exact role the AI should assume for the analysis.")
    else:
        persona = selected_persona

    input_method = st.sidebar.radio("Input Method:", ["CSV File Upload", "YouTube Link", "Raw Text Paste", "Telegram Dump (JSON)", "Reddit Native (OSINT)"], horizontal=True)
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

    elif input_method == "Reddit Native (OSINT)":
        with st.form("reddit_form"):
            st.info("Extracts the latest posts directly from any public Subreddit without API keys.")
            sub_input = st.text_input("Subreddit Name", placeholder="e.g., worldnews, conspiracy, italia")
            limit = st.number_input("Max Posts to Scrape", min_value=10, max_value=100, value=50, step=10)
            submitted = st.form_submit_button("Extract Intel")
            
            if submitted and sub_input:
                with st.spinner(f"Infiltrating r/{sub_input}..."):
                    rdf = scrape_reddit_native(sub_input, limit)
                    if rdf is not None:
                        rdf = detect_bot_activity(rdf)
                        st.session_state['data_store'][input_method]['df'] = rdf
                        st.session_state['data_store'][input_method]['analyzed'] = None 
                        st.session_state['data_store'][input_method]['summary'] = None
                        st.success(f"Intercepted {len(rdf)} posts from r/{sub_input}!")
                    else:
                        st.error("Failed to extract data. Make sure the subreddit is public and spelled correctly.")

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
                            
                            ita_stops = {
                                "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da", "in", "con", "su", "per", "tra", "fra", 
                                "e", "o", "ma", "se", "perché", "non", "che", "chi", "cui", "mi", "ti", "ci", "vi", "si", "ho", "ha", "hanno", 
                                "è", "sono", "sei", "siamo", "siete", "era", "erano", "c'è", "ne", "al", "allo", "alla", "ai", "agli", "alle", 
                                "del", "dello", "della", "dei", "degli", "delle", "dal", "dallo", "dalla", "dai", "dagli", "dalle", "nel", "nello", 
                                "nella", "nei", "negli", "nelle", "sul", "sullo", "sulla", "sui", "sugli", "sulle", "questo", "quello", "più", 
                                "anche", "tutto", "tutti", "solo", "fare", "fatto", "essere", "stato", "poi", "quando", "molto", "così", "quindi", 
                                "dopo", "invece", "ancora", "già", "senza", "sempre", "ora", "qui", "lì", "quale", "cosa", "loro", "come"
                            }
                            custom_stops = set(STOPWORDS).union(ita_stops)
                            
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
                    ans = analyze_fallacies(row['content'], api_key=key, context_info=context_input, persona=persona, target_lang=report_language)
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
                            st.download_button("Download Strategy", action_plan, "strategy.txt", type="primary")
                    else: st.warning("Please select at least one comment in the table above.")

                st.markdown("---")
                st.markdown("---")
                st.subheader("Psy-Ops Target Profiler")
                st.caption("Select a specific User/Agent to run a deep behavioral analysis on their messaging patterns.")
                
                c_psy1, c_psy2 = st.columns([1, 2])
                with c_psy1:
                    unique_users = adf['agent_id'].unique()
                    selected_target = st.selectbox("Select Target (Agent ID)", unique_users)
                    run_psyops = st.button("Run Behavioral Profile", type="primary")
                
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

                    st.markdown("---")
                    st.caption("Propaganda Warfare Matrix")
                    if 'sophistication' in adf.columns and 'hostile_intent' in adf.columns:
                        # Clean data for plotting
                        mat_df = adf[['agent_id', 'sophistication', 'hostile_intent', 'propaganda_tactic', 'content']].copy()
                        mat_df['sophistication'] = pd.to_numeric(mat_df['sophistication'], errors='coerce').fillna(0)
                        mat_df['hostile_intent'] = pd.to_numeric(mat_df['hostile_intent'], errors='coerce').fillna(0)
                        
                        chart_matrix = alt.Chart(mat_df).mark_circle(size=80, opacity=0.7).encode(
                            x=alt.X('sophistication:Q', title='Sophistication (Complexity)', scale=alt.Scale(domain=[0, 10])),
                            y=alt.Y('hostile_intent:Q', title='Hostile Intent (Malice)', scale=alt.Scale(domain=[0, 10])),
                            color=alt.Color('propaganda_tactic:N', title='Tactic', scale=alt.Scale(scheme='category10')),
                            tooltip=['agent_id', 'propaganda_tactic', 'sophistication', 'hostile_intent', 'content']
                        ).interactive().properties(height=400)
                        
                        xrule = alt.Chart(pd.DataFrame({'x': [5]})).mark_rule(color='gray', strokeDash=[3,3]).encode(x='x')
                        yrule = alt.Chart(pd.DataFrame({'y': [5]})).mark_rule(color='gray', strokeDash=[3,3]).encode(y='y')
                        
                        st.altair_chart(chart_matrix + xrule + yrule, use_container_width=True)
                        st.info("Top-Right Quadrant (High Sophistication + High Hostility) contains the most dangerous Information Warfare operators.")

                with st.expander("Target-Narrative Network Graph", expanded=False):
                    net_fig = plot_network_graph(adf)
                    if net_fig:
                        st.plotly_chart(net_fig, use_container_width=True)
                    else:
                        st.caption("Not enough Target/Topic data to build a network.")

                st.markdown("---")

                # --- PATIENT ZERO & CHIEF PROPAGANDIST DETECTOR ---
                st.markdown("---")
                st.subheader("High-Value Targets (HVT) Identification")
                st.caption("Algorithmic detection of the most dangerous actors based on toxicity, logical fallacies, and network impact (likes).")
                
                if 'likes' in adf.columns and 'aggression' in adf.columns and 'has_fallacy' in adf.columns:
                    # Filter for toxic/fallacious content
                    toxic_subset = adf[(adf['has_fallacy'] == True) | (adf['aggression'] > 6)]
                    
                    if not toxic_subset.empty:
                        # Group by user to calculate their total destructive impact
                        hvt_stats = toxic_subset.groupby('agent_id').agg(
                            total_impact=('likes', 'sum'),
                            avg_toxicity=('aggression', 'mean'),
                            fallacy_count=('has_fallacy', 'sum')
                        ).reset_index()
                        
                        # Calculate a custom "Threat Score" formula
                        hvt_stats['threat_score'] = (hvt_stats['total_impact'] * 0.5) + (hvt_stats['avg_toxicity'] * 10) + (hvt_stats['fallacy_count'] * 20)
                        hvt_stats = hvt_stats.sort_values(by='threat_score', ascending=False).head(3)
                        
                        c_hvt1, c_hvt2, c_hvt3 = st.columns(3)
                        hvt_cols = [c_hvt1, c_hvt2, c_hvt3]
                        
                        for idx, (_, row) in enumerate(hvt_stats.iterrows()):
                            with hvt_cols[idx]:
                                st.error(f"🎯 **Target Rank #{idx+1}**")
                                st.markdown(f"**Agent ID:** `{row['agent_id']}`")
                                st.markdown(f"**Threat Score:** {int(row['threat_score'])}")
                                st.caption(f"Fallacies: {row['fallacy_count']} | Impact: {row['total_impact']} likes")
                    else:
                        st.success("No High-Value Targets detected (Network is relatively healthy).")
                else:
                    st.caption("Insufficient data to calculate High-Value Targets (missing likes or aggression metrics).")
                
                # --- SOCKPUPPET DETECTOR (STYLOMETRY) ---
                st.markdown("---")
                st.subheader("🎭 Sockpuppet & Troll Farm Detector")
                st.caption("Stylometric analysis: Identifies different Agent IDs that exhibit the exact same writing style, grammar flaws, and punctuation tics (indicating a single operator managing multiple fake accounts).")
                
                if st.button("Run Stylometric AI Scan", type="primary"):
                    with st.spinner("Analyzing linguistic patterns and neuro-linguistic tics across users..."):
                        # Group comments by user
                        user_comments = adf.groupby('agent_id')['content'].apply(lambda x: " | ".join(x)).reset_index()
                        
                        if len(user_comments) > 1:
                            # Take a maximum of 15 users to prevent token explosion
                            sample_users = user_comments.head(15).to_string(index=False)
                            
                            sockpuppet_prompt = f"""
                            You are an elite Stylometric and Linguistic Forensics AI.
                            Analyze the following database of users and their comments.
                            
                            TASK: Identify "Sockpuppets" (Troll Farm activity). Find instances where 2 or more DIFFERENT 'agent_id' show the EXACT same distinct writing style, neuro-linguistic tics, repetitive punctuation, or grammatical errors.
                            
                            DATA:
                            {sample_users}
                            
                            CRITICAL RULE: Write your final report in the SAME LANGUAGE as the comments provided.
                            If no sockpuppets are found, state that the linguistic variance is natural. If you find suspects, name the 'agent_id' pairs and explain the linguistic tics that link them.
                            """
                            try:
                                client = genai.Client(api_key=key)
                                sock_res = client.models.generate_content(model='gemini-2.5-flash', contents=sockpuppet_prompt)
                                st.warning("### Stylometric Audit Report")
                                st.markdown(sock_res.text)
                            except Exception as e:
                                st.error(f"Stylometric Scan Error: {str(e)}")
                        else:
                            st.info("Not enough unique users to run a stylometric comparison.")

                # --- DARK TRIAD PSYCHOLOGICAL PROFILING ---
                st.markdown("---")
                st.subheader("Dark Triad Profiling")
                st.caption("Analyzes the text of a specific agent to detect Narcissism, Machiavellianism, and Psychopathy.")
                
                # Dropdown to select the target
                target_agent = st.selectbox("Select Target for Psych-Profile:", adf['agent_id'].unique())
                
                if st.button("Run Psychological Profile", type="primary"):
                    with st.spinner(f"Psychoanalyzing {target_agent}..."):
                        # Gather all comments from this user
                        target_comments = " | ".join(adf[adf['agent_id'] == target_agent]['content'].tolist())
                        
                        dt_prompt = f"""
                        You are an elite Psychological Profiler.
                        Analyze the psychological profile of the author based on these texts:
                        "{target_comments}"
                        
                        Evaluate the Dark Triad traits from 0 to 100.
                        CRITICAL RULE: Return ONLY a JSON in this exact format:
                        {{
                            "narcissism": 50,
                            "machiavellianism": 50,
                            "psychopathy": 50,
                            "analysis": "Short clinical explanation of the profile."
                        }}
                        """
                        try:
                            client = genai.Client(api_key=key)
                            dt_res = client.models.generate_content(model='gemini-2.5-flash', contents=dt_prompt)
                            dt_json = extract_json(dt_res.text)
                            
                            if dt_json:
                                st.write(f"**Clinical Analysis:** {dt_json.get('analysis', 'N/A')}")
                                
                                # Plot the Radar Chart
                                df_dt = pd.DataFrame(dict(
                                    r=[dt_json.get('narcissism', 0), dt_json.get('machiavellianism', 0), dt_json.get('psychopathy', 0)],
                                    theta=['Narcissism', 'Machiavellianism', 'Psychopathy']
                                ))
                                fig_dt = px.line_polar(df_dt, r='r', theta='theta', line_close=True, range_r=[0,100], title=f"Dark Triad Signature: {target_agent}")
                                fig_dt.update_traces(fill='toself', line_color='red')
                                st.plotly_chart(fig_dt, use_container_width=True)
                        except Exception as e:
                            st.error(f"Profiling Error: {e}")

                # --- CROSS-ENTITY RESOLUTION (GLOBAL MEMORY CHECK) ---
                st.markdown("---")
                st.subheader("Cross-Entity Resolution (Global Memory Check)")
                st.caption("Cross-referencing currently analyzed actors and topics with secret documents from the Deep Oracle.")
                
                if 'global_entities' in st.session_state and len(st.session_state['global_entities']) > 0:
                    found_matches = set()
                    for _, row in adf.iterrows():
                        # Check agent_id against memory
                        agent = str(row.get('agent_id', '')).lower()
                        if agent in st.session_state['global_entities']: found_matches.add(row['agent_id'])
                        
                        # Check target entity against memory
                        target = str(row.get('target', '')).lower()
                        if target in st.session_state['global_entities']: found_matches.add(row['target'])
                        
                    if found_matches:
                        st.error(f"🚨 **CRITICAL INTELLIGENCE MATCH FOUND!**")
                        st.warning(f"The following entities in this social dataset were ALSO found in the classified documents (Module 6): **{', '.join(found_matches)}**")
                    else:
                        st.success("No cross-entity matches found in Global Memory.")
                else:
                    st.info("Global Memory is empty. Extract a Power Graph in Module 6 to populate it.")
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

                # --- INTERACTIVE DYNAMIC TIMELINE ---
                st.markdown("---")
                st.subheader("Dynamic Timeline Filter")
                view = adf.copy()
                if 'timestamp' in view.columns:
                    # Add the new column directly to 'view' instead of 'adf'
                    view['parsed_time'] = pd.to_datetime(view['timestamp'], errors='coerce')
                    valid_time_df = view.dropna(subset=['parsed_time'])
                    
                    if not valid_time_df.empty:
                        min_time = valid_time_df['parsed_time'].min()
                        max_time = valid_time_df['parsed_time'].max()
                        
                        # Show the timeline if there is at least some time difference
                        if min_time != max_time:
                            # Slider with hours and minutes instead of just days
                            selected_times = st.slider(
                                "Filter Analysis by Time Window", 
                                min_value=min_time.to_pydatetime(), 
                                max_value=max_time.to_pydatetime(), 
                                value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
                                format="YYYY-MM-DD HH:mm"
                            )
                            # Now 'view' has the 'parsed_time' column, so this will work perfectly
                            view = view[(view['parsed_time'] >= selected_times[0]) & (view['parsed_time'] <= selected_times[1])]
                            
                            timeline_data = valid_time_df[(valid_time_df['parsed_time'] >= selected_times[0]) & (valid_time_df['parsed_time'] <= selected_times[1])]
                            
                            # Group data by exact timestamp to show minute-by-minute peaks
                            agg_time = timeline_data.groupby('parsed_time')['aggression'].mean().reset_index()
                            agg_time.columns = ['Time', 'Avg Aggression']
                            
                            if not agg_time.empty:
                                line_chart = alt.Chart(agg_time).mark_area(opacity=0.4, color='red').encode(
                                    x='Time:T', y=alt.Y('Avg Aggression:Q', scale=alt.Scale(domain=[0, 10]))
                                ) + alt.Chart(agg_time).mark_line(color='darkred', point=True).encode(
                                    x='Time:T', y='Avg Aggression:Q', tooltip=['Time:T', 'Avg Aggression:Q']
                                )
                                st.altair_chart(line_chart, use_container_width=True)
                        else:
                            st.caption("Not enough time variance to build a dynamic timeline (all messages share the exact same second).")
                    else:
                        st.caption("Not enough valid dates to build a dynamic timeline.")
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
                        c1.write(status)
                        bot_msg = f" | ⚠️ BOT SUSPECT: {r.get('bot_reason')}" if r.get('is_bot') else ""
                        emotion_tag = f" | {r.get('primary_emotion', '')}" if r.get('primary_emotion') else ""
                        likes_tag = f" | 👍 {r.get('likes', 0)}"
                        arch_tag = f" | 🎭 {r.get('archetype', 'User')}"
                        c2.caption(f"**User:** {r.get('agent_id', 'User')} | **Agg:** {r.get('aggression')}/10 {likes_tag}{emotion_tag}{arch_tag}{bot_msg}")
                        # --- UNIVERSAL TRANSLATOR UI ---
                        translated = str(r.get('translated_text', ''))
                        if translated and translated.lower() not in ["none", "nan", "", "null"] and translated != r['content']:
                            st.info(f"**[Translated]:** \"{translated}\"\n\n*Original:* {r['content']}")
                        else:
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
                    st.download_button("Download Full Excel Report", excel_data, "RAP_Intelligence_Report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
                with c_down2:
                    if st.button("Generate PDF Report", type="primary"):
                        pdf_bytes = generate_pdf_report(adf, summary_text=summary_text) 
                        st.download_button("Download PDF", pdf_bytes, "RAP_Executive_Report.pdf", "application/pdf", type="primary")
                
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
elif mode == "3. Cognitive Editor (Text/Image/Audio/Video)":
    st.header("3. Cognitive Editor & Fact-Checker")
    st.caption("Upload Text, Images (Memes/Screenshots), Audio clips or Videos for deep inspection.")
    
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
                original_img = Image.open(f)

                # --- OPSEC: BIOMETRIC ANONYMIZATION ---
                censor_faces = st.checkbox("Apply OPSEC Face Censor (Auto-Anonymize)", value=False, help="Automatically detects and redacts human faces before analysis to protect identities.")
                
                if censor_faces:
                    media_inp, face_count = anonymize_faces(original_img)
                    if face_count > 0:
                        st.success(f"Classified: {face_count} identities redacted.")
                    else:
                        st.caption("No faces detected by the algorithm.")
                else:
                    media_inp = original_img

                media_type = "image"
                st.image(media_inp, caption="Evidence Image", use_container_width=True)
                
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
                
                # --- OPSEC IMAGE SCRUBBER ---
                st.markdown("---")
                st.markdown("#### OPSEC: Metadata Scrubber")
                st.caption("Remove all invisible EXIF data (GPS, Device Info) before sharing this evidence.")
                
                # Function to strip EXIF using PIL
                def strip_exif(image):
                    data = list(image.getdata())
                    image_without_exif = Image.new(image.mode, image.size)
                    image_without_exif.putdata(data)
                    return image_without_exif

                clean_image = strip_exif(media_inp)
                
                # Save to bytes for downloading
                img_byte_arr = io.BytesIO()
                clean_image.save(img_byte_arr, format='PNG')
                clean_bytes = img_byte_arr.getvalue()

                st.download_button(
                    label="Download Sanitized Image (Zero EXIF)",
                    data=clean_bytes,
                    file_name=f"RAP_Sanitized_Evidence_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                    mime="image/png",
                    type="primary"
                )

                # --- ELA FORENSICS VISUALIZER ---
                st.markdown("#### Error Level Analysis (ELA)")
                st.caption("Detects digital manipulation (Photoshop/copy-paste). Artificially inserted elements will glow significantly brighter or have a completely different texture/color in the ELA map compared to the rest of the image.")
                
                with st.spinner("Generating ELA Map..."):
                    ela_img = perform_ela(media_inp)
                    if ela_img:
                        st.image(ela_img, caption="ELA Heatmap (Look for glowing/inconsistent edges)", use_container_width=True)
                    else:
                        st.warning("Could not generate ELA for this image format.")

        elif inp_type == "Audio (Voice Intel)":
            f = st.file_uploader("Upload Audio", type=['mp3', 'wav', 'm4a'])
            if f:
                media_inp = f.read() 
                media_type = "audio"
                st.audio(media_inp, format='audio/mp3')
                
                # --- VISUAL WAVEFORM ---
                with st.spinner("Generating waveform..."):
                    waveform_img = generate_audio_waveform(media_inp)
                    if waveform_img:
                        st.image(waveform_img, use_container_width=True)
                        st.caption("Inspect the waveform for unnatural silences or frequency clipping (typical of AI voice clones).")
        
        elif inp_type == "Video (Deepfake Scan)":
            text_inp = st.text_area("Video Context (Optional)", placeholder="What is this video claiming?", height=100)
            
            # Modify the base prompt to tell it it's looking at a storyboard grid
            text_inp = "CRITICAL INSTRUCTION: You are looking at a Forensic Storyboard grid of a video, not a single photo. Compare the physical consistency of the matter between the different frames to find AI glitches.\n\n" + str(text_inp)

            f = st.file_uploader("Upload Video (Max 50MB)", type=['mp4', 'mov'])
            if f:
                raw_video_bytes = f.read()
                st.video(raw_video_bytes)
                
                with st.spinner("Extracting forensic storyboard (Nuclear Option)..."):
                    storyboard_bytes = create_video_storyboard(raw_video_bytes, num_frames=12)
                    if storyboard_bytes:
                        st.success("✅ Forensic storyboard extracted invisibly for the AI.")
                        # We pass the grid to Gemini making it believe it's a single image!
                        media_inp = storyboard_bytes
                        media_type = "image"
                    else:
                        st.error("Failed to extract frames.")
            else:
                st.info("Please upload an MP4 or MOV file to start the forensic analysis.")
            
        go = st.button("Analyze, Sanitize & Scan AI", use_container_width=True, type="primary")

    with c2:
        st.subheader("Output (Analysis & Sanitize)")
        if go:
            if media_inp or text_inp:
                with st.spinner(f"Processing with Gemini ({inp_type})..."):
                    ret = cognitive_rewrite(text_inp, key, media_inp, media_type, target_lang=report_language)
                    
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
                        if inp_type == "Video (Deepfake Scan)" and isinstance(v_timeline, list) and len(v_timeline) > 0:
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
                            st.error(f"🛑 Issue Detected: **{ret.get('fallacy_type', 'System Processing Error')}**")
                            st.metric("Aggression Level", f"{ret.get('aggression', 0)}/10")
                            st.warning(f"**Analysis:** {ret.get('explanation', 'No details available.')}")
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
                            st.markdown("#### Voice & Tone Analysis")
                            if stress > 65:
                                st.error(f"**Voice Stress Score: {stress}%** (High emotion, anger, or panic detected in prosody)")
                            else:
                                st.success(f"**Voice Stress Score: {stress}%** (Calm, controlled, or neutral tone)")
                            st.markdown("---")
                        if media_type in ["image", "video"] and ret.get('shadow_geolocation') and ret.get('shadow_geolocation') != "Not applicable":
                            st.markdown("#### Shadow Geolocation (Visual GeoINT)")
                            st.info(f"**AI Visual Deduction:** {ret['shadow_geolocation']}")
                            st.markdown("---")
                        
                        # --- GHOST READER (OCR FORENSICS) ---
                        if ret.get('ocr_extraction') and ret.get('ocr_extraction').lower() not in ["none", "n/a", "", "null"]:
                            st.markdown("#### Ghost Reader (OCR Extraction)")
                            st.caption("AI optical scan of texts hidden in the media (signs, screens, documents).")
                            st.info(f"{ret['ocr_extraction']}")
                            st.markdown("---")

                        # --- AUDIO/VIDEO TRANSCRIPTION ---
                        if media_type in ["audio", "video"] and ret.get('transcript'):
                            st.markdown("#### Audio Transcription (Speech-to-Text)")
                            st.info(f"{ret['transcript']}")
                            st.markdown("---")
                        st.markdown("#### Fact Checker (Claims to Verify)")
                        facts = ret.get('facts', [])
                        if facts:
                            for f in facts: st.write(f"- {f}")
                        else:
                            st.caption("No specific factual claims found.")

                        # --- FORENSIC DOSSIER EXPORT (WITH HASHING) ---
                        st.markdown("---")
                        
                        # Calculate cryptographic hash for Chain of Custody
                        file_hash = calculate_sha256(media_inp) if media_type in ['audio', 'video'] else "N/A (Image Object or Text)"
                        
                        # Compile the plain text report for investigators
                        dossier_text = f"RAP FORENSIC DOSSIER\n"
                        dossier_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                        dossier_text += f"Target Media Type: {inp_type}\n"
                        dossier_text += f"Digital Fingerprint (SHA-256): {file_hash}\n"
                        dossier_text += f"Chain of Custody Status: SECURED\n"
                        dossier_text += f"{'='*40}\n\n"
                        
                        dossier_text += f"[1] AI GENERATION SCAN\n"
                        dossier_text += f"AI Probability Score: {ai_prob}%\n"
                        dossier_text += f"Forensic Analysis: {ret.get('ai_analysis', 'N/A')}\n\n"
                        
                        if ret.get('transcript'):
                            dossier_text += f"[2] AUDIO TRANSCRIPTION\n"
                            dossier_text += f"Transcript: {ret.get('transcript')}\n\n"
                        
                        dossier_text += f"[3] LOGIC & FALLACY CHECK\n"
                        dossier_text += f"Issue Detected: {ret.get('fallacy_type', 'None')}\n"
                        dossier_text += f"Aggression Level: {ret.get('aggression', 0)}/10\n"
                        dossier_text += f"Explanation: {ret.get('explanation', 'N/A')}\n\n"
                        
                        if ret.get('shadow_geolocation') and ret.get('shadow_geolocation') != "Not applicable":
                            dossier_text += f"[4] SHADOW GEOLOCATION (VISUAL OSINT)\n"
                            dossier_text += f"Deduction: {ret.get('shadow_geolocation')}\n\n"
                            
                        if ret.get('facts'):
                            dossier_text += f"[5] EXTRACTED CLAIMS TO VERIFY\n"
                            for f_claim in ret.get('facts'):
                                dossier_text += f"- {f_claim}\n"
                        
                        st.download_button(
                            label="Download Forensic Dossier (TXT)",
                            data=dossier_text.encode('utf-8'),
                            file_name=f"RAP_Forensic_Dossier_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain",
                            type="primary"
                        )
            else:
                st.warning("Please provide input.")

# ==========================================
# MODULE 4: COMPARISON TEST (A/B TESTING)
# ==========================================
elif mode == "4. Comparison Test (A/B Testing)":
    st.header("4. Universal Arena (A/B Testing)")
    st.caption("Compare data from YouTube, CSV, or Raw Text to find the most aggressive narratives.")
    
    col_a, col_b = st.columns(2)
    
    def load_arena_data(key_prefix, column):
        with column:
            in_type = st.radio(f"Input {key_prefix}", ["YouTube Link", "CSV Upload", "Raw Text Paste"], horizontal=True, key=f"r_{key_prefix}")
            df = None
            if in_type == "YouTube Link":
                url = st.text_input(f"YouTube URL {key_prefix}", key=f"url_{key_prefix}")
                limit = st.number_input(f"Comments to Fetch {key_prefix}", 10, 1000, 50, key=f"lim_{key_prefix}")
                if st.button(f"Scrape {key_prefix}", key=f"btn_yt_{key_prefix}"):
                    with st.spinner(f"Scraping YouTube {key_prefix}..."):
                        df = scrape_youtube_comments(url, limit)
            elif in_type == "CSV Upload":
                f = st.file_uploader(f"Upload CSV {key_prefix}", type="csv", key=f"f_{key_prefix}")
                if f: df = normalize_dataframe(pd.read_csv(f))
            else:
                txt = st.text_area(f"Paste Data {key_prefix}", height=100, key=f"t_{key_prefix}")
                if st.button(f"Process Text {key_prefix}", key=f"btn_txt_{key_prefix}"):
                    if txt: df = parse_raw_paste(txt)
                    
            if df is not None:
                df = detect_bot_activity(df)
                if 'Select' not in df.columns: df.insert(0, "Select", False)
                st.session_state['data_store']['Arena'][f'df_{key_prefix.lower()}'] = df
                st.success(f"Loaded {len(df)} items in {key_prefix}.")
            
    st.subheader("Contender A (Left)")
    load_arena_data("A", col_a)
    st.subheader("Contender B (Right)")
    load_arena_data("B", col_b)

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
        
        st.markdown("---")
        st.subheader("Tactical Visual Comparison")
        
        # 1. Base Aggression Chart
        res_df_a['Source'] = 'Contender A'
        res_df_b['Source'] = 'Contender B'
        combined = pd.concat([res_df_a, res_df_b])
        
        chart_agg = alt.Chart(combined).mark_bar().encode(
            x=alt.X('Source', title=None),
            y=alt.Y('mean(aggression)', title='Avg Aggression'),
            color=alt.Color('Source', scale=alt.Scale(domain=['Contender A', 'Contender B'], range=['#3b82f6', '#f97316'])),
            tooltip=['Source', 'mean(aggression)']
        ).properties(height=200)
        st.altair_chart(chart_agg, use_container_width=True)
        
        # 2. Advanced Divergence Charts (Emotions & Fallacies)
        c_vis1, c_vis2 = st.columns(2)
        
        with c_vis1:
            st.caption("Emotional Spectrum Clash")
            if 'primary_emotion' in combined.columns:
                emo_combined = combined.groupby(['Source', 'primary_emotion']).size().reset_index(name='Count')
                chart_emo = alt.Chart(emo_combined).mark_bar().encode(
                    x=alt.X('primary_emotion:N', title='Emotion', sort='-y'),
                    y=alt.Y('Count:Q', title='Frequency'),
                    color=alt.Color('Source:N', scale=alt.Scale(domain=['Contender A', 'Contender B'], range=['#3b82f6', '#f97316']), legend=None),
                    xOffset='Source:N',
                    tooltip=['Source', 'primary_emotion', 'Count']
                ).properties(height=300)
                st.altair_chart(chart_emo, use_container_width=True)
                
        with c_vis2:
            st.caption("Weaponized Fallacies")
            if 'fallacy_type' in combined.columns:
                fal_combined = combined[combined['has_fallacy'] == True].groupby(['Source', 'fallacy_type']).size().reset_index(name='Count')
                if not fal_combined.empty:
                    chart_fal = alt.Chart(fal_combined).mark_bar().encode(
                        x=alt.X('fallacy_type:N', title='Fallacy Type', sort='-y'),
                        y=alt.Y('Count:Q', title='Frequency'),
                        color=alt.Color('Source:N', scale=alt.Scale(domain=['Contender A', 'Contender B'], range=['#3b82f6', '#f97316'])),
                        xOffset='Source:N',
                        tooltip=['Source', 'fallacy_type', 'Count']
                    ).properties(height=300)
                    st.altair_chart(chart_fal, use_container_width=True)
                else:
                    st.success("No fallacies detected in either contender.")

        # --- AI NARRATIVE CLASH BRIEFING ---
        st.markdown("---")
        st.subheader("The Oracle: Narrative Clash Assessment")
        st.caption("Force the AI to analyze the differing psychological and tactical profiles of the two contenders.")
        
        if st.button("Generate Clash Briefing", type="primary"):
            with st.spinner("Analyzing psychological divergence..."):
                top_emotions_a = res_df_a['primary_emotion'].value_counts().head(3).to_dict() if 'primary_emotion' in res_df_a else "N/A"
                top_emotions_b = res_df_b['primary_emotion'].value_counts().head(3).to_dict() if 'primary_emotion' in res_df_b else "N/A"
                top_fallacies_a = res_df_a['fallacy_type'].value_counts().head(3).to_dict() if 'fallacy_type' in res_df_a else "N/A"
                top_fallacies_b = res_df_b['fallacy_type'].value_counts().head(3).to_dict() if 'fallacy_type' in res_df_b else "N/A"
                
                # Estraiamo un piccolo campione di testo per far capire la lingua all'IA
                sample_a = res_df_a['content'].iloc[0] if not res_df_a.empty else ""
                sample_b = res_df_b['content'].iloc[0] if not res_df_b.empty else ""
                
                clash_prompt = f"""
                You are an elite Information Warfare and Psychological Operations Analyst.
                Compare these two opposing groups (Contender A vs Contender B).
                
                CONTENDER A:
                - Avg Aggression: {agg_a:.1f}/10
                - Dominant Emotions: {top_emotions_a}
                - Primary Logical Fallacies used: {top_fallacies_a}
                - Sample Text: "{sample_a[:200]}"
                
                CONTENDER B:
                - Avg Aggression: {agg_b:.1f}/10
                - Dominant Emotions: {top_emotions_b}
                - Primary Logical Fallacies used: {top_fallacies_b}
                - Sample Text: "{sample_b[:200]}"
                
                TASK: Write a sharp, tactical "Narrative Clash Briefing" (strictly max 150 words). 
                Do not just list the numbers. Explain *HOW* their manipulation strategies differ. 
                Conclude by stating which group poses a higher risk of inciting real-world polarization.
                
                CRITICAL RULE: Detect the language of the "Sample Text" provided above. You MUST write the ENTIRE briefing strictly in that SAME language (e.g., if the sample is in Italian, the whole output must be in Italian).
                """
                try:
                    client = genai.Client(api_key=key)
                    clash_res = client.models.generate_content(model='gemini-2.5-flash', contents=clash_prompt)
                    st.warning("### Tactical Clash Report")
                    st.markdown(clash_res.text)
                except Exception as e:
                    st.error(f"Failed to generate briefing: {e}")

        with st.expander("Detailed Comparison Data"):
            st.dataframe(combined[['Source', 'content', 'aggression', 'primary_emotion', 'fallacy_type', 'is_bot']])

        # --- PDF EXPORT FOR ARENA (A/B TESTING) ---
        st.markdown("---")
        st.subheader("Export Battle Report")
        st.caption("Generate a comparative PDF dossier detailing the metrics between Contender A and Contender B.")
        
        if st.button("Generate Arena PDF", type="primary"):
            with st.spinner("Compiling the comparative report..."):
                pdf = PDFReport()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                
                # Arena specific header
                pdf.set_font("Helvetica", 'B', 14)
                pdf.cell(0, 10, "Comparative Dossier: Narrative A/B Testing", 0, 1)
                pdf.ln(5)
                
                # Section 1: Contender A Profile
                pdf.set_font("Helvetica", 'B', 12)
                pdf.cell(0, 10, "Contender A Profile:", 0, 1)
                pdf.set_font("Helvetica", '', 11)
                pdf.cell(0, 8, f"- Analyzed Items: {len(res_df_a)}", 0, 1)
                pdf.cell(0, 8, f"- Average Aggression: {agg_a:.1f}/10", 0, 1)
                pdf.cell(0, 8, f"- Detected Bots: {bots_a}", 0, 1)
                pdf.cell(0, 8, f"- Logical Fallacies Flagged: {fallacy_a}", 0, 1)
                pdf.ln(5)

                # Section 2: Contender B Profile
                pdf.set_font("Helvetica", 'B', 12)
                pdf.cell(0, 10, "Contender B Profile:", 0, 1)
                pdf.set_font("Helvetica", '', 11)
                pdf.cell(0, 8, f"- Analyzed Items: {len(res_df_b)}", 0, 1)
                pdf.cell(0, 8, f"- Average Aggression: {agg_b:.1f}/10", 0, 1)
                pdf.cell(0, 8, f"- Detected Bots: {bots_b}", 0, 1)
                pdf.cell(0, 8, f"- Logical Fallacies Flagged: {fallacy_b}", 0, 1)
                pdf.ln(5)
                
                # Section 3: Tactical Conclusion
                pdf.set_font("Helvetica", 'B', 12)
                pdf.cell(0, 10, "Tactical Conclusion:", 0, 1)
                pdf.set_font("Helvetica", 'I', 11)
                
                if agg_a > agg_b + 1.5:
                    conclusion = "Contender A exhibits a significantly higher level of hostility and aggression. It poses a greater immediate risk for polarization."
                elif agg_b > agg_a + 1.5:
                    conclusion = "Contender B exhibits a significantly higher level of hostility and aggression. It poses a greater immediate risk for polarization."
                else:
                    conclusion = "Both contenders present comparable levels of aggression. The threat level is balanced between the two narratives."
                    
                if bots_a > bots_b:
                    conclusion += f" Furthermore, Contender A shows higher signs of inauthentic/bot activity (+{delta_bots})."
                elif bots_b > bots_a:
                    conclusion += f" Furthermore, Contender B shows higher signs of inauthentic/bot activity (+{abs(delta_bots)})."

                pdf.multi_cell(0, 6, conclusion.encode('latin-1', 'replace').decode('latin-1'))
                
                pdf_bytes_arena = bytes(pdf.output())
                
            st.download_button(
                label="Download Arena Report (PDF)", 
                data=pdf_bytes_arena, 
                file_name=f"RAP_Arena_Match_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", 
                mime="application/pdf", 
                type="primary"
            )

# ==========================================
# MODULE 5: LIVE RADAR (RSS/REDDIT)
# ==========================================
elif mode == "5. Live Radar (RSS/Reddit)":
    st.header("5. Live Radar (Crisis Alert System)")
    st.caption("Monitor live RSS feeds or subreddits to intercept escalating disinformation and aggression in real-time.")
    
    if "GEMINI_API_KEY" in st.secrets: key = st.secrets["GEMINI_API_KEY"]
    else: key = st.text_input("API Key", type="password")
    
    c_radar1, c_radar2, c_radar3 = st.columns([2, 1, 1])
    with c_radar1:
        feed_url = st.text_input("Enter RSS Feed, Subreddit, or News Keyword", placeholder="E.g., ansa, bbc, repubblica, reddit.com/r/worldnews")
    with c_radar2:
        world_languages = [
            "English", "Italiano", "Español", "Français", "Deutsch", "Português",
            "Русский (Russian)", "العربية (Arabic)", "中文 (Chinese)", 
            "日本語 (Japanese)", "فارسی (Persian)", "हिन्दी (Hindi)", 
            "한국어 (Korean)", "Türkçe (Turkish)"
        ]
        news_region = st.selectbox("Search Language/Region", world_languages, index=1)
    with c_radar3:
        max_entries = st.number_input("Entries", 5, 50, 15)
        fetch_btn = st.button("Step 1: Fetch Feed", type="primary", use_container_width=True)
        # --- DEFCON CYBER SCAN BUTTON ---
        defcon_btn = st.button("🚨 DEFCON Cyber Scan", type="primary", use_container_width=True)
    # --- AUTOMATED ALERT CONFIGURATION ---
    with st.expander("Automated Alert Configuration (Webhook)"):
        alert_webhook = st.text_input("Webhook URL", placeholder="https://hooks.slack.com/services/...", help="If Aggression exceeds the threshold, an alert payload will be dispatched here.")
        
        # Quick guide for generating Webhooks
        with st.expander("ℹ️ How to get a Webhook URL (Discord / Slack)"):
            st.markdown("""
            **For Discord:**
            1. Open your Server Settings -> **Integrations** -> **Webhooks**.
            2. Click **New Webhook**, name it "RAP Sentinel", select a channel, and click **Copy Webhook URL**.
            
            **For Slack:**
            1. Go to your Workspace Administration -> **Manage Apps**.
            2. Search for **Incoming Webhooks** and add it to your desired channel.
            3. Copy the generated **Webhook URL**.
            """)
            
        alert_threshold = st.slider("Trigger Alert Threshold (Aggression)", min_value=1.0, max_value=10.0, value=8.0, step=0.5, help="Dispatch alerts only if the average aggression exceeds this level.")
        st.caption(f"Note: System will dispatch emergency protocols if average aggression spikes above {alert_threshold}/10.")

    if (fetch_btn and feed_url) or defcon_btn:
        with st.spinner("Intercepting live feed..."):
            try:
                user_input = ""
                
                if defcon_btn:
                    actual_url = "https://feeds.feedburner.com/TheHackersNews"
                    st.toast("DEFCON Protocol Activated: Intercepting Global Cyber Threats...", icon="🚨")
                else:
                    user_input = feed_url.strip().lower()
                    actual_url = feed_url.strip()
                    
                news_shortcuts = {
                    "ansa": "https://www.ansa.it/sito/ansait_rss.xml",
                    "repubblica": "https://www.repubblica.it/rss/homepage/rss2.0.xml",
                    "corriere": "http://xml2.corriereobjects.it/rss/homepage.xml",
                    "bbc": "http://feeds.bbci.co.uk/news/world/rss.xml",
                    "cnn": "http://rss.cnn.com/rss/edition.rss",
                    "nytimes": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"
                }
                
                if user_input and user_input in news_shortcuts:
                    actual_url = news_shortcuts[user_input]
                elif user_input and "reddit.com/r/" in user_input and not user_input.endswith(".rss"):
                    if actual_url.endswith("/"): actual_url = actual_url[:-1]
                    actual_url += "/new/.rss"
                elif not actual_url.startswith("http"):
                    safe_query = actual_url.replace(" ", "%20")
                    
                    lang_map = {
                        "English": "hl=en-US&gl=US&ceid=US:en",
                        "Italiano": "hl=it&gl=IT&ceid=IT:it",
                        "Español": "hl=es&gl=ES&ceid=ES:es",
                        "Français": "hl=fr&gl=FR&ceid=FR:fr",
                        "Deutsch": "hl=de&gl=DE&ceid=DE:de",
                        "Português": "hl=pt-BR&gl=BR&ceid=BR:pt-419",
                        "Русский (Russian)": "hl=ru&gl=RU&ceid=RU:ru",
                        "العربية (Arabic)": "hl=ar&gl=EG&ceid=EG:ar",
                        "中文 (Chinese)": "hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
                        "日本語 (Japanese)": "hl=ja&gl=JP&ceid=JP:ja",
                        "فارسی (Persian)": "hl=fa&gl=IR&ceid=IR:fa",
                        "हिन्दी (Hindi)": "hl=hi&gl=IN&ceid=IN:hi",
                        "한국어 (Korean)": "hl=ko&gl=KR&ceid=KR:ko",
                        "Türkçe (Turkish)": "hl=tr&gl=TR&ceid=TR:tr"
                    }
                    
                    region_params = lang_map.get(news_region, "hl=it&gl=IT&ceid=IT:it")
                    actual_url = f"https://news.google.com/rss/search?q={safe_query}&{region_params}"
                
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
            st.markdown("**Running Threat Analysis (Crisis Protocol)...**")
            
            for i, (_, row) in enumerate(subset.iterrows()):
                crisis_prompt = f"""
                You are an Early Warning Crisis & Threat Intelligence AI. 
                Analyze the following news excerpt. 
                CRITICAL INSTRUCTION: Do NOT evaluate the linguistic tone. Journalists write neutrally. 
                Instead, evaluate the INHERENT CRISIS LEVEL, CONFLICT, and POLARIZATION of the EVENT described.
                
                Scoring Guide (0 to 10):
                0-3 (Stable): Routine news, tech, sports, diplomacy, standard political processes.
                4-6 (Elevated): Economic trouble, political scandals, diplomatic friction, peaceful protests.
                7-8 (Critical): Riots, extreme societal polarization, localized violence, severe systemic threats.
                9-10 (Extreme): War, terrorism, mass casualties, global geopolitical collapse.
                
                News Text: "{row['content']}"
                
                CRITICAL LANGUAGE RULE: Write the "reasoning" strictly in the SAME LANGUAGE as the "News Text" above.
                CRITICAL GEO RULE: Extract the 3-letter ISO Alpha-3 country code of the primary nation involved in the event (e.g., "USA", "ITA", "RUS", "UKR"). If it's a global event or unknown, use "GLO".
                
                Respond ONLY with a JSON in this exact format:
                {{"aggression": [number from 0 to 10], "reasoning": "Brief 1-sentence tactical justification in the target language", "iso_country": "XXX"}}
                """
                try:
                    client = genai.Client(api_key=key)
                    response = client.models.generate_content(model='gemini-2.5-flash', contents=crisis_prompt)
                    parsed = extract_json(response.text)
                    
                    if parsed:
                        parsed['explanation'] = parsed.get('reasoning', 'No reasoning provided.')
                        if parsed.get('aggression', 0) >= 7.0:
                            parsed['has_fallacy'] = True
                            parsed['fallacy_type'] = "High Crisis Alert"
                        else:
                            parsed['has_fallacy'] = False
                    else:
                        parsed = {"aggression": 0, "explanation": "Analysis failed.", "has_fallacy": False}
                except Exception as e:
                    parsed = {"aggression": 0, "explanation": f"API Error: {str(e)}", "has_fallacy": False}
                
                res.append(sanitize_response(parsed))
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
        
        soglia = alert_threshold if 'alert_threshold' in locals() else 8.0
        
        if avg_aggression >= soglia:
            st.error(f"🚨 **CRITICAL CRISIS ALERT:** The current feed has exceeded the aggression threshold ({avg_aggression:.1f} >= {soglia}).")
            # --- DISPATCH ALERT ---
            if 'alert_webhook' in locals() and alert_webhook:
                st.toast("Dispatching emergency alert to configured channels...", icon="🚨")
                st.success(f"📧 **Automated Alert Dispatched!** Sent payload to {alert_webhook} at {datetime.now().strftime('%H:%M:%S')}")
        elif avg_aggression >= (soglia - 2.0):
            st.warning(f"⚠️ **ELEVATED TENSION:** The feed is showing signs of polarization and is approaching the alert threshold.")
        # --- GEO-INT CHOROPLETH MAP ---
        if 'iso_country' in analyzed_radar.columns:
            st.markdown("---")
            st.subheader("Geopolitical Crisis Map")
            
            map_data = analyzed_radar[~analyzed_radar['iso_country'].isin(['GLO', 'None', 'Unknown', ''])].copy()
            
            if not map_data.empty:
                geo_stats = map_data.groupby('iso_country').agg(
                    News_Count=('iso_country', 'count'),
                    Avg_Tension=('aggression', 'mean')
                ).reset_index()
                
                fig_map = px.choropleth(
                    geo_stats, 
                    locations="iso_country", 
                    color="Avg_Tension",
                    hover_name="iso_country",
                    hover_data=["News_Count"],
                    color_continuous_scale=px.colors.sequential.Reds,
                    range_color=(0, 10),
                    title="Live Global Tension Heatmap"
                )
                fig_map.update_layout(
                    geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular', bgcolor='rgba(0,0,0,0)'),
                    paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.caption("No specific geographical targets identified in the current feed.")
        st.markdown("---")
        
        st.subheader("Live Feed Feedbacks")
        for _, r in analyzed_radar.iterrows():
            with st.container(border=True):
                st.caption(f"🕒 {r['timestamp']} | [Source Link]({r['link']}) | **Agg:** {r['aggression']}/10")
                st.write(r['content'][:300] + "...")
                if r['has_fallacy']:
                    st.error(f"🛑 **{r['fallacy_type']}**: {r['explanation']}")
                else:
                    st.success(f"✅ {r.get('explanation', 'Clear.')}")
        st.divider()
        st.subheader("Export Radar Intel")
        st.caption("Download the monitored items for your intelligence reports.")
        
        # Prepare CSV data
        export_df = analyzed_radar.drop(columns=['Select'], errors='ignore')
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        
        # Prepare PDF data
        with st.spinner("Generating Threat PDF Report..."):
            pdf_bytes_radar = generate_pdf_report(
                analyzed_radar, 
                summary_text=f"LIVE RADAR ALERT.\nFeed monitored: {feed_url}\nAverage Aggression Level: {avg_aggression:.1f}/10\nTotal items flagged for issues: {len(analyzed_radar[analyzed_radar['has_fallacy']==True])}"
            ) 
            
        # Display buttons side by side
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            st.download_button(
                label="Download Alert Report (CSV)",
                data=csv_data,
                file_name=f"RAP_Radar_Alert_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary",
            )
        with c_btn2:
            st.download_button(
                label="Download Threat Report (PDF)",
                data=pdf_bytes_radar,
                file_name=f"RAP_Radar_Threat_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                type="primary",
            )
        # ==========================================
        # --- LIVE SENTINEL MODE (AUTONOMOUS) ---
        # ==========================================
        st.divider()
        st.subheader("Live Sentinel Mode (Autonomous)")
        st.caption("Leave this dashboard open. The system will continuously fetch, analyze, and dispatch alerts automatically via Webhook when new critical events occur.")

        c_live1, c_live2 = st.columns([1, 2])
        with c_live1:
            live_toggle = st.toggle("Activate Live Sentinel", key="live_radar_toggle")
        with c_live2:
            if live_toggle:
                refresh_rate = st.slider("Scan Interval (Seconds)", 15, 600, 60, help="How often to refresh the feed in background.")

        if live_toggle:
            if not feed_url:
                st.warning("Please enter a Keyword or Feed URL at the top first.")
            else:
                live_placeholder = st.empty()
                
                with live_placeholder.container():
                    st.info(f"🟢 **SENTINEL ACTIVE** | Target: **{feed_url}** | Next scan in {refresh_rate}s")
                    
                    with st.spinner("Scanning for new events..."):
                        # --- 1. SILENT FETCH ---
                        user_input = feed_url.strip().lower()
                        actual_url = feed_url.strip()
                        news_shortcuts = {
                            "ansa": "https://www.ansa.it/sito/ansait_rss.xml",
                            "repubblica": "https://www.repubblica.it/rss/homepage/rss2.0.xml",
                            "corriere": "http://xml2.corriereobjects.it/rss/homepage.xml",
                            "bbc": "http://feeds.bbci.co.uk/news/world/rss.xml",
                            "cnn": "http://rss.cnn.com/rss/edition.rss",
                            "nytimes": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"
                        }
                        
                        if user_input in news_shortcuts:
                            actual_url = news_shortcuts[user_input]
                        elif "reddit.com/r/" in user_input and not user_input.endswith(".rss"):
                            if actual_url.endswith("/"): actual_url = actual_url[:-1]
                            actual_url += "/new/.rss"
                        elif not actual_url.startswith("http"):
                            safe_query = actual_url.replace(" ", "%20")
                            
                            # Comprehensive mapping for the 14 global languages supported by RAP
                            lang_map = {
                                "English": "hl=en-US&gl=US&ceid=US:en",
                                "Italiano": "hl=it&gl=IT&ceid=IT:it",
                                "Español": "hl=es&gl=ES&ceid=ES:es",
                                "Français": "hl=fr&gl=FR&ceid=FR:fr",
                                "Deutsch": "hl=de&gl=DE&ceid=DE:de",
                                "Português": "hl=pt-BR&gl=BR&ceid=BR:pt-419",
                                "Русский (Russian)": "hl=ru&gl=RU&ceid=RU:ru",
                                "العربية (Arabic)": "hl=ar&gl=EG&ceid=EG:ar",
                                "中文 (Chinese)": "hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
                                "日本語 (Japanese)": "hl=ja&gl=JP&ceid=JP:ja",
                                "فارسی (Persian)": "hl=fa&gl=IR&ceid=IR:fa",
                                "हिन्दी (Hindi)": "hl=hi&gl=IN&ceid=IN:hi",
                                "한국어 (Korean)": "hl=ko&gl=KR&ceid=KR:ko",
                                "Türkçe (Turkish)": "hl=tr&gl=TR&ceid=TR:tr"
                            }
                            # Fetch the correct region parameters based on user selection
                            region_params = lang_map.get(news_region if 'news_region' in locals() else "Italiano", "hl=it&gl=IT&ceid=IT:it")
                            actual_url = f"https://news.google.com/rss/search?q={safe_query}&{region_params}"
                            
                        feed = feedparser.parse(actual_url)
                        
                        # --- 2. FILTER ONLY NEW ITEMS ---
                        new_items = []
                        for entry in feed.entries[:int(max_entries)]:
                            link = entry.get('link', '')
                            # Check if the news item is already in memory
                            if link not in st.session_state['seen_radar_links']:
                                clean_summary = re.sub(r'<[^>]+>', '', entry.get('summary', ''))
                                new_items.append({
                                    'title': entry.get('title', ''),
                                    'content': f"TITLE: {entry.get('title', '')}\nCONTENT: {clean_summary[:500]}",
                                    'link': link
                                })
                        
                        # --- 3. AI ANALYSIS & ALERTING ---
                        if new_items:
                            st.warning(f"🚨 Intercepted {len(new_items)} new updates! Running Threat Analysis...")
                            
                            live_agg_scores = []
                            for item in new_items:
                                crisis_prompt = f"""
                                You are an Early Warning Crisis AI. Evaluate the INHERENT CRISIS LEVEL of the EVENT.
                                Scoring (0 to 10): 0-3 (Stable), 4-6 (Elevated), 7-8 (Critical), 9-10 (Extreme).
                                News: "{item['content']}"
                                
                                CRITICAL LANGUAGE RULE: Write the "reasoning" strictly in the SAME LANGUAGE as the "News".
                                CRITICAL GEO RULE: Extract the 3-letter ISO Alpha-3 country code (e.g., "USA", "ITA"). If global/unknown, use "GLO".
                                
                                Respond ONLY with a JSON: {{"aggression": [0-10], "reasoning": "Brief reason in the target language", "iso_country": "XXX"}}
                                """
                                try:
                                    client = genai.Client(api_key=key)
                                    response = client.models.generate_content(model='gemini-2.5-flash', contents=crisis_prompt)
                                    parsed = extract_json(response.text)
                                    score = parsed.get('aggression', 0) if parsed else 0
                                except:
                                    score = 0
                                
                                live_agg_scores.append(score)
                                # Register the link so it won't be analyzed again
                                st.session_state['seen_radar_links'].add(item['link'])
                                
                            avg_live_agg = sum(live_agg_scores) / len(live_agg_scores) if live_agg_scores else 0
                            st.metric("New Items Crisis Index", f"{avg_live_agg:.1f}/10")
                            
                            # --- 4. DISPATCH REAL WEBHOOK ---
                            soglia = alert_threshold if 'alert_threshold' in locals() else 8.0
                            if avg_live_agg >= soglia:
                                st.error(f"🚨 ALERT THRESHOLD BREACHED ({avg_live_agg:.1f} >= {soglia})! DISPATCHING PAYLOAD...")
                                
                                # Dispatch the actual notification
                                if 'alert_webhook' in locals() and alert_webhook:
                                    try:
                                        payload = {
                                            "content": f"🚨 **RAP CRISIS ALERT** 🚨\n**Target:** {feed_url}\n**Crisis Level:** {avg_live_agg:.1f}/10\n**New Events Detected:** {len(new_items)}\n**Status:** Critical tension detected in live feed. Open dashboard for details.",
                                            "text": f"🚨 *RAP CRISIS ALERT* 🚨\n*Target:* {feed_url}\n*Crisis Level:* {avg_live_agg:.1f}/10\n*New Events Detected:* {len(new_items)}\n*Status:* Critical tension detected in live feed. Open dashboard for details."
                                        }
                                        # Fire the Webhook HTTP POST request (works for Slack and Discord)
                                        requests.post(alert_webhook, json=payload, timeout=5)
                                        st.success("✅ Webhook delivered successfully to your channel!")
                                    except Exception as e:
                                        st.error(f"Failed to send Webhook: {e}")
                            else:
                                st.success("Tension is below threshold. No alert dispatched.")
                        else:
                            st.success(f"No new events detected since last scan. Standing by.")
                            
                    # --- 5. VISUAL COUNTDOWN TIMER ---
                    timer_ui = st.empty()
                    for remaining in range(refresh_rate, 0, -1):
                        timer_ui.info(f"**Radar in Standby.** Next autonomous sweep in: **{remaining}s**")
                        time.sleep(1)
                        
                    st.rerun()
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
            
            # --- PII SANITIZER ---
            c_san1, c_san2 = st.columns([1, 3])
            with c_san1:
                if st.button("Sanitize Document (Redact PII)", help="Hides Emails, IPs, IBANs, and Phones before chatting with the Oracle"):
                    with st.spinner("Sanitizing sensitive data..."):
                        clean_text = sanitize_pii(st.session_state['doc_full_text'])
                        st.session_state['doc_full_text'] = clean_text
                        st.success("Document successfully sanitized (CIA Blackout Protocol)!")
            
            # --- DOWNLOAD REDACTED DOSSIER ---
            st.download_button(
                label="⬛ Download Redacted Dossier (TXT)",
                data=st.session_state['doc_full_text'].encode('utf-8'),
                file_name=f"RAP_Classified_Redacted_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                type="primary"
            )
            # ------------------------------
            
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
                                
                                # --- GLOBAL MEMORY REGISTRATION ---
                                # Save extracted highly-sensitive entities into the cross-module memory
                                for rel in graph_json.get('relations', []):
                                    if rel.get('source'): st.session_state['global_entities'].add(str(rel['source']).strip().lower())
                                    if rel.get('target'): st.session_state['global_entities'].add(str(rel['target']).strip().lower())
                                st.success(f"Registered {len(st.session_state['global_entities'])} entities into Global Memory for cross-referencing.")
                            else:
                                st.error("Not enough clear relationships found to build a graph.")
                        except Exception as e:
                            st.error(f"Graph Extraction Error: {str(e)}")
            st.divider()
            # ---------------------------------------------

            # --- CONTRADICTION & LOOPHOLE SCANNER ---
            with st.expander("Deep Scan: Contradictions & Loopholes", expanded=False):
                st.caption("Force the AI to audit the entire document specifically looking for logical contradictions, legal loopholes, or unfulfilled claims.")
                if st.button("Run Audit Scan", type="primary"):
                    with st.spinner("Auditing document for contradictions..."):
                        audit_prompt = f"""
                        You are a ruthless Forensic Auditor and Legal Analyst.
                        Scan the following document specifically for:
                        1. Internal Contradictions (e.g., Chapter 1 says X, Chapter 4 says the opposite of X).
                        2. Loopholes or ambiguous clauses that could be exploited.
                        3. Hidden risks or highly controversial statements.
                        
                        CRITICAL RULE: For every point you make, you MUST cite the [PAGE X] reference. If the document is flawless, state that no contradictions were found.
                        
                        DOCUMENT TEXT:
                        {st.session_state['doc_full_text']}
                        """
                        try:
                            client = genai.Client(api_key=key)
                            audit_res = client.models.generate_content(model='gemini-2.0-flash', contents=audit_prompt)
                            st.warning("### ⚠️ Forensic Audit Report")
                            st.markdown(audit_res.text)
                            
                            # Add the audit to the oracle history so the user can continue chatting about it
                            st.session_state.doc_oracle_history.append({"role": "assistant", "content": f"**[AUTOMATED AUDIT REPORT]**\n\n{audit_res.text}"})
                        except Exception as e:
                            st.error(f"Audit Error: {str(e)}")
                            
            # --- MULTI-AGENT DEBATE (RED TEAM vs BLUE TEAM) ---
            with st.expander("Multi-Agent War Room (Stress Test)", expanded=False):
                st.caption("Deploy two opposing AI agents to debate the document. The Red Team attacks it, the Blue Team defends it.")
                debate_topic = st.text_input("Debate Focus (e.g., 'Security flaws', 'Ethical implications', 'Legal robustness'):", placeholder="What should the agents fight about?")
                
                if st.button("Initiate Agent Debate", type="primary"):
                    if not debate_topic:
                        st.warning("Please enter a debate focus.")
                    else:
                        st.markdown(f"### 🔴 Red Team vs 🔵 Blue Team: *{debate_topic}*")
                        
                        # AGENT 1: RED TEAM (ATTACKER)
                        with st.spinner("🔴 Red Team is analyzing vulnerabilities..."):
                            red_prompt = f"You are the RED TEAM (Aggressive Attacker). Criticize this document focusing on: '{debate_topic}'. Find flaws, risks, and weaknesses. Be ruthless. Use maximum 150 words.\n\nCRITICAL LANGUAGE RULE: You MUST write your ENTIRE response strictly in the SAME LANGUAGE as the '{debate_topic}' and the DOCUMENT.\n\nDOCUMENT: {st.session_state['doc_full_text'][:20000]}"
                            client = genai.Client(api_key=key)
                            red_res = client.models.generate_content(model='gemini-2.5-flash', contents=red_prompt).text
                            st.error(f"**🔴 Red Team Attack:**\n{red_res}")
                        
                        # AGENT 2: BLUE TEAM (DEFENDER)
                        with st.spinner("🔵 Blue Team is formulating a defense..."):
                            blue_prompt = f"You are the BLUE TEAM (Steadfast Defender). Read the RED TEAM's attack below regarding the document. Counter their arguments based on the text. Minimize risks and defend the document. Use maximum 150 words.\n\nCRITICAL LANGUAGE RULE: You MUST write your ENTIRE response strictly in the SAME LANGUAGE as the RED TEAM ATTACK.\n\nRED TEAM ATTACK:\n{red_res}\n\nDOCUMENT: {st.session_state['doc_full_text'][:20000]}"
                            blue_res = client.models.generate_content(model='gemini-2.5-flash', contents=blue_prompt).text
                            st.info(f"**🔵 Blue Team Defense:**\n{blue_res}")
                            
                        # AGENT 3: THE JUDGE
                        with st.spinner("⚖️ The Judge is deliberating..."):
                            judge_prompt = f"You are the IMPARTIAL JUDGE. Read the debate between Red and Blue. Who won? Provide a final 2-sentence verdict on the actual risk level of the document regarding '{debate_topic}'.\n\nCRITICAL LANGUAGE RULE: You MUST write your ENTIRE response strictly in the SAME LANGUAGE as the RED and BLUE debate.\n\nRED:\n{red_res}\n\nBLUE:\n{blue_res}"
                            judge_res = client.models.generate_content(model='gemini-2.5-flash', contents=judge_prompt).text
                            st.success(f"**⚖️ Final Verdict:**\n{judge_res}")
                            
                        st.session_state.doc_oracle_history.append({"role": "assistant", "content": f"**[WAR ROOM DEBATE: {debate_topic}]**\n\n**🔴 RED:** {red_res}\n\n**🔵 BLUE:** {blue_res}\n\n**⚖️ JUDGE:** {judge_res}"})
            st.divider()
            
            # --- CHRONO-INTELLIGENCE (TIMELINE EXTRACTION) ---
            with st.expander("Chrono-Intelligence (Event Timeline)", expanded=False):
                st.caption("Extract every date, deadline, and historical event mentioned in the document and plot them chronologically.")
                if st.button("Generate Timeline", type="primary"):
                    with st.spinner("Extracting chronological data..."):
                        timeline_prompt = f"""
                        You are a Chronological Intelligence AI. 
                        Read the document and extract every significant event that has a clear Date or Year associated with it.
                        
                        CRITICAL RULE: Return ONLY a JSON in this exact format:
                        {{ "events": [ {{"date": "YYYY-MM-DD" (or just YYYY), "event": "Short description of what happened"}} ] }}
                        
                        DOCUMENT TEXT:
                        {st.session_state['doc_full_text'][:30000]}
                        """
                        try:
                            client = genai.Client(api_key=key)
                            time_res = client.models.generate_content(model='gemini-2.0-flash', contents=timeline_prompt)
                            time_json = extract_json(time_res.text)
                            
                            if time_json and 'events' in time_json and len(time_json['events']) > 0:
                                t_df = pd.DataFrame(time_json['events'])
                                # Attempt to parse dates for plotting
                                t_df['ParsedDate'] = pd.to_datetime(t_df['date'], errors='coerce')
                                t_df = t_df.dropna(subset=['ParsedDate']).sort_values('ParsedDate')
                                
                                if not t_df.empty:
                                    # Create a beautiful Plotly Timeline
                                    fig_time = px.scatter(
                                        t_df, x="ParsedDate", y=[1]*len(t_df), text="event",
                                        title="Document Chronological Timeline",
                                        labels={"ParsedDate": "Timeline", "y": ""},
                                        height=300
                                    )
                                    fig_time.update_traces(
                                        mode="markers+text", 
                                        textposition="top center",
                                        marker=dict(size=12, color="red")
                                    )
                                    fig_time.update_yaxes(showticklabels=False, showgrid=False, zeroline=True, zerolinecolor="gray")
                                    st.plotly_chart(fig_time, use_container_width=True)
                                    
                                    # Show table
                                    st.dataframe(t_df[['date', 'event']], hide_index=True, use_container_width=True)
                                    
                                    st.session_state.doc_oracle_history.append({"role": "assistant", "content": f"**[TIMELINE EXTRACTED]**\nFound {len(t_df)} chronological events."})
                                else:
                                    st.error("Could not parse valid dates from the document.")
                            else:
                                st.warning("No clear chronological events found in the text.")
                        except Exception as e:
                            st.error(f"Timeline Extraction Error: {str(e)}")
            st.divider()

            # --- PROJECT SPIDERWEB (KNOWLEDGE GRAPH) ---
            with st.expander("Project Spiderweb (Entity Network)", expanded=False):
                st.caption("Maps the hidden connections (financial, political, social) between entities in the document.")
                if st.button("Generate Spiderweb Graph", type="primary"):
                    with st.spinner("Extracting entities and calculating network topology..."):
                        net_prompt = f"""
                        Extract the main entities (People, Companies, Countries, Organizations) and how they are connected.
                        Limit to the top 15 most critical relationships to keep the graph readable.
                        
                        CRITICAL RULE: Return ONLY a JSON in this exact format:
                        {{"edges": [{{"source": "Entity 1", "target": "Entity 2", "label": "e.g., funded, opposes, controls"}}]}}
                        
                        DOCUMENT TEXT:
                        {st.session_state['doc_full_text'][:20000]}
                        """
                        try:
                            client = genai.Client(api_key=key)
                            net_res = client.models.generate_content(model='gemini-2.5-flash', contents=net_prompt)
                            net_json = extract_json(net_res.text)
                            
                            if net_json and 'edges' in net_json and len(net_json['edges']) > 0:
                                # Build the Graph using NetworkX
                                G = nx.DiGraph()
                                for edge in net_json['edges']:
                                    G.add_edge(edge['source'], edge['target'], label=edge['label'])
                                
                                # Calculate positions for the nodes
                                pos = nx.spring_layout(G, seed=42)
                                
                                # Create edges for Plotly
                                edge_x, edge_y, edge_text = [], [], []
                                for edge in G.edges(data=True):
                                    x0, y0 = pos[edge[0]]
                                    x1, y1 = pos[edge[1]]
                                    edge_x.extend([x0, x1, None])
                                    edge_y.extend([y0, y1, None])
                                    edge_text.append(edge[2]['label'])
                                
                                # Create nodes for Plotly
                                node_x = [pos[node][0] for node in G.nodes()]
                                node_y = [pos[node][1] for node in G.nodes()]
                                
                                # Draw the Plotly Figure
                                fig_net = go.Figure()
                                # Add lines (Edges)
                                fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'), hoverinfo='none', mode='lines'))
                                # Add dots (Nodes)
                                fig_net.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), textposition="top center", marker=dict(size=25, color='cyan', line=dict(width=2, color='DarkSlateGrey')), hoverinfo='text'))
                                
                                fig_net.update_layout(title="Entity Relationship Network", showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=500)
                                st.plotly_chart(fig_net, use_container_width=True)
                            else:
                                st.warning("Not enough relationships found to build a network.")
                        except Exception as e:
                            st.error(f"Spiderweb Generation Error: {e}")
            st.divider()

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
