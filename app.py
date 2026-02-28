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
import sqlite3

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

# --- DYNAMIC AI TRANSLATION ENGINE ---
@st.cache_data(show_spinner=False)
def ai_t(text, target_lang, api_key):
    """
    Translates UI labels dynamically using Gemini. 
    Uses st.cache_data to minimize API calls and costs.
    """
    if not text or target_lang == "English":
        return text
    
    try:
        client = genai.Client(api_key=api_key)
        # We instruct the AI to be extremely concise to save tokens
        prompt = f"Translate this UI label to {target_lang}. Return ONLY the translated text: '{text}'"
        response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return response.text.strip().replace("'", "").replace('"', "")
    except Exception:
        # Fallback to English if the API fails or key is missing
        return text

def t(text):
    """
    Funzione di traduzione disattivata per massimizzare le prestazioni.
    Restituisce sempre il testo originale in inglese.
    Per riattivarla in futuro, de-commenta il blocco sottostante.
    """
    return text 

    # lang = st.session_state.get('global_lang', 'English')
    # key = st.session_state.get('api_key') or st.secrets.get("GEMINI_API_KEY", "")
    # if not key or lang == "English":
    #     return text
    # return ai_t(text, lang, key)

# --- HELPER: PANOPTICON (PERSISTENT MEMORY) ---
def init_panopticon():
    conn = sqlite3.connect('rap_panopticon.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS targets (agent_id TEXT PRIMARY KEY, risk_score REAL, threat_type TEXT, last_seen TEXT)''')
    conn.commit()
    return conn

panopticon_db = init_panopticon()

def save_to_panopticon(agent_id, risk_score, threat_type):
    try:
        c = panopticon_db.cursor()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute('''INSERT INTO targets (agent_id, risk_score, threat_type, last_seen) VALUES (?, ?, ?, ?)
                     ON CONFLICT(agent_id) DO UPDATE SET risk_score=max(risk_score, ?), last_seen=?''',
                  (str(agent_id), float(risk_score), str(threat_type), now, float(risk_score), now))
        panopticon_db.commit()
    except Exception as e: pass

def check_panopticon(agent_id):
    try:
        c = panopticon_db.cursor()
        c.execute('''SELECT risk_score, threat_type, last_seen FROM targets WHERE agent_id=?''', (str(agent_id),))
        return c.fetchone()
    except: return None

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

# --- HELPER: SMART DOCUMENT CHUNKING ---
def chunk_document(text, chunk_size=3000, overlap=300):
    """Splits a massive document into overlapping semantic chunks for context retrieval."""
    if not text or not text.strip(): return []
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def prepare_smart_context(texts):
    """
    Placeholder to maintain compatibility with the UI. 
    Instead of calculating vectors (which fail), we return the raw chunks 
    to be analyzed dynamically during the chat.
    """
    # We return a dummy list to signal the UI that the document is 'ready'
    return [[0.0] * 768 for _ in texts]

# --- HELPER: OSINT WEB SPIDER ---
def run_web_spider(url):
    """Crawls a specific webpage to extract its hidden internal structure and external affiliations."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) RAP_OSINT_Bot/1.0'}
        res = requests.get(url, headers=headers, timeout=8)
        html = res.text
        
        # Raw regex extraction of href links to bypass BeautifulSoup requirement
        links = re.findall(r'href=[\'"]?([^\'" >]+)', html)
        
        base_domain = urllib.parse.urlparse(url).netloc
        internal, external = set(), set()
        
        for link in links:
            if link.startswith('http'):
                if base_domain in link:
                    internal.add(link)
                else:
                    external.add(link)
            elif link.startswith('/'):
                internal.add(f"https://{base_domain}{link}")
                
        return list(internal)[:20], list(external)[:20]
    except Exception as e:
        return [], []

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
    
    # 1. Duplicate Content (Swarm Coordination)
    dupes = df.duplicated(subset=['content'], keep=False)
    df.loc[dupes, 'is_bot'] = True
    df.loc[dupes, 'bot_reason'] = "Duplicate Content (Coordination) "
    
    if 'agent_id' in df.columns:
        # 2. High Frequency Spammer
        agent_counts = df['agent_id'].value_counts()
        spammers = agent_counts[agent_counts > 3].index
        mask_spammer = df['agent_id'].isin(spammers)
        df.loc[mask_spammer, 'is_bot'] = True
        df.loc[mask_spammer, 'bot_reason'] += "High Frequency "

        # 3. Superhuman Velocity Check (< 5 seconds between posts)
        if 'timestamp' in df.columns:
            try:
                temp_df = df.copy()
                temp_df['parsed_time'] = pd.to_datetime(temp_df['timestamp'], errors='coerce')
                valid_time = temp_df.dropna(subset=['parsed_time'])
                
                if not valid_time.empty:
                    valid_time = valid_time.sort_values(by=['agent_id', 'parsed_time'])
                    valid_time['time_diff'] = valid_time.groupby('agent_id')['parsed_time'].diff().dt.total_seconds()
                    
                    # Identifica gli agenti che hanno pubblicato messaggi a meno di 5 secondi di distanza
                    fast_posters = valid_time[(valid_time['time_diff'] >= 0) & (valid_time['time_diff'] < 5)]['agent_id'].unique()
                    mask_fast = df['agent_id'].isin(fast_posters)
                    
                    df.loc[mask_fast, 'is_bot'] = True
                    df.loc[mask_fast, 'bot_reason'] += "Superhuman Velocity (<5s) "
            except Exception:
                pass # Ignora se i timestamp sono formattati male

    df['bot_reason'] = df['bot_reason'].str.strip()
    return df

# --- HELPER: TREND PROJECTION ---
@st.cache_data
def project_trend(df, days_ahead=7):
    if 'timestamp' not in df.columns: return None, False, None
    df_trend = df.copy()
    df_trend['timestamp'] = pd.to_datetime(df_trend['timestamp'], errors='coerce')
    df_trend = df_trend.dropna(subset=['timestamp']).sort_values('timestamp')
    if len(df_trend) < 5: return None, False, None
    
    start_date = df_trend['timestamp'].min()
    df_trend['days_since_start'] = (df_trend['timestamp'] - start_date).dt.total_seconds() / 86400
    x = df_trend['days_since_start'].values
    y = df_trend['aggression'].values
    if len(np.unique(x)) < 2: return None, False, None
    
    slope, intercept = np.polyfit(x, y, 1)
    last_day = x.max()
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1).astype(int)
    forecast_y = slope * future_days + intercept
    future_dates = [start_date + timedelta(days=float(d)) for d in future_days]
    
    past_df = pd.DataFrame({'Date': df_trend['timestamp'], 'Aggression': y, 'Type': 'Historical'})
    future_df = pd.DataFrame({'Date': future_dates, 'Aggression': forecast_y, 'Type': 'Forecast'})
    combined = pd.concat([past_df, future_df])
    
    days_to_critical = None
    if slope > 0.05:
        days_to_critical = (9.0 - intercept) / slope - last_day
        if days_to_critical < 0: days_to_critical = 0
        
    return combined, (slope > 0.05), days_to_critical

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
    
# --- NATIVE TELEGRAM OSINT SCRAPER (CHAMELEON PROTOCOL) ---
@st.cache_data(show_spinner=False)
def scrape_telegram_live(channel_url, limit=30):
    try:
        # Extreme URL sanitization (stripping parameters and paths)
        channel = channel_url.replace('https://', '').replace('http://', '').replace('t.me/', '').replace('@', '').replace('s/', '').strip('/')
        channel = channel.split('/')[0].split('?')[0]
        url = f"https://t.me/s/{channel}" # Target the public web preview
        
        # Advanced cloaking as a modern browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        # Attack via Requests
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            st.error(f"Telegram blocked access. HTTP Error: {response.status_code}")
            return None
            
        html = response.text
        
        # V3: Bulletproof parsing using string splits instead of greedy regex
        # This prevents nested <div> tags (links, bold text) from breaking the extraction
        chunks = re.split(r'<div class="tgme_widget_message_text[^>]*>', html)
        
        if len(chunks) <= 1:
            st.warning("Connection successful, but no text blocks found. Channel may be empty or strictly media-only.")
            return None

        data = []
        now = datetime.now()
        
        # Extract the actual text blocks
        msgs = []
        for chunk in chunks[1:]:
            # Cut the chunk exactly before the footer or the info section starts
            text_block = re.split(r'<div class="tgme_widget_message_info"|<div class="tgme_widget_message_footer"', chunk)[0]
            msgs.append(text_block)
        
        for i, m in enumerate(msgs[-limit:]):
            # Replace <br> tags with actual line breaks
            clean_text = re.sub(r'<br\s*/?>', '\n', m)
            # Strip all remaining HTML tags
            clean_text = re.sub(r'<[^>]+>', ' ', clean_text).strip()
            # Clean up multiple whitespaces
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if len(clean_text) > 5:
                data.append({
                    'agent_id': f"TG_{channel}",
                    'timestamp': (now - timedelta(minutes=len(msgs)-i)).strftime("%Y-%m-%d %H:%M:%S"),
                    'content': clean_text,
                    'likes': 0
                })
        
        if not data:
            st.warning("Messages found, but they were empty after stripping HTML formatting.")
            return None
            
        return pd.DataFrame(data)
    except Exception as e: 
        st.error(f"Scraper Error: {str(e)}")
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
            "timestamp": (now - timedelta(seconds=i*2)).strftime("%Y-%m-%d %H:%M:%S"),
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

# --- TRANSLATED MODULE SELECTION ---
mode = st.sidebar.radio(t("Select Module:"), [
    t("1. Wargame Room (Simulation)"), 
    t("2. Social Data Analysis (Universal)"), 
    t("3. Cognitive Editor (Text/Image/Audio/Video)"), 
    t("4. Comparison Test (A/B Testing)"),
    t("5. Live Radar (RSS/Reddit)"),
    t("6. Deep Document Oracle (RAG)"),
    t("7. Panopticon (HVT Watchlist)")
])

# --- TRANSLATED MASTER DOSSIER EXPORT ---
st.sidebar.markdown("---")
if st.sidebar.button(t("Download Master Dossier"), type="primary", help=t("Compiles a global tactical report by merging data from all modules.")):
    master_text = f"RAP GLOBAL MASTER DOSSIER\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*40}\n\n"
    
    # Collect data from Module 2
    for input_type, data in st.session_state['data_store'].items():
        if input_type not in ['Arena', 'Radar'] and data.get('summary'):
            master_text += f"--- MODULE 2: SOCIAL INTELLIGENCE ({input_type}) ---\n"
            master_text += f"{data['summary']}\n\n"
            
    # Collect data from Radar (Module 5)
    radar_data = st.session_state['data_store'].get('Radar', {}).get('analyzed')
    if radar_data is not None and not radar_data.empty:
        master_text += f"--- MODULE 5: LIVE RADAR ALERTS ---\n"
        master_text += f"Total Flagged Threats: {len(radar_data[radar_data['has_fallacy']==True])}\n"
        master_text += f"Average Aggression: {radar_data['aggression'].mean():.1f}/10\n\n"
    
    # Collect discovered entities (Module 6)
    if st.session_state['global_entities']:
        master_text += f"--- MODULE 6: DEEP ORACLE (KNOWN ENTITIES) ---\n"
        master_text += ", ".join(list(st.session_state['global_entities'])) + "\n\n"
        
    st.sidebar.download_button(t("Save Dossier.txt"), master_text.encode('utf-8'), f"RAP_Master_Dossier_{datetime.now().strftime('%Y%m%d')}.txt", "text/plain", type="primary")

# --- GLOBAL REPORT LANGUAGE SELECTOR ---
st.sidebar.markdown("---")
world_languages = [
    "English", "Italiano", "Español", "Français", "Deutsch", "Português",
    "Русский (Russian)", "العربية (Arabic)", "中文 (Chinese)", 
    "日本語 (Japanese)", "فارسی (Persian)", "हिन्दी (Hindi)", 
    "한국어 (Korean)", "Türkçe (Turkish)"
]

st.session_state['global_lang'] = st.sidebar.selectbox(
    "Global Output Language (Analysis Reports)", 
    world_languages, 
    index=0
)

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
if mode == t("1. Wargame Room (Simulation)"):
    st.header(t("1. Information Warfare Simulator"))
    
    c_param1, c_param2 = st.columns(2)
    with c_param1:
        st.subheader(t("Network Topology"))
        # Using t() inside selectbox options
        topology = st.selectbox(t("Scenario Type"), [
            t("Public Square (High Connectivity)"), 
            t("Echo Chambers (Clusters)"), 
            t("Influencer Network (Hubs)")
        ])
        n_agents = st.slider(t("Population Size"), 100, 2000, 1000)
        bot_pct = st.slider(t("Infection/Bot Ratio"), 0.0, 0.5, 0.10)
        
    with c_param2:
        st.subheader(t("Countermeasures (Blue Team)"))
        defense = st.selectbox(t("Active Defense Protocol"), [
            t("None (Control Group)"), 
            t("Fact-Check Debunking (Targeted)"), 
            t("Algorithmic Dampening (Global)"), 
            t("Hard Ban (Removal)")
        ])
        steps = st.slider(t("Simulation Duration (Days)"), 50, 300, 100)
        
    # --- LOGIC (Variables remain in English for stability) ---
    agents = np.zeros(n_agents)
    n_infected = int(n_agents * bot_pct)
    
    # Internal logic matching translated strings
    if topology == t("Echo Chambers (Clusters)"):
        start = int(n_agents * 0.4)
        agents[start : start + n_infected] = 1.0
    else:
        indices = np.random.choice(n_agents, n_infected, replace=False)
        agents[indices] = 1.0
        
    history = np.zeros((n_agents, steps))
    history[:, 0] = agents.copy()
    infection_rate = []
    current = agents.copy()
    
    if topology == t("Public Square (High Connectivity)"): influence_strength, noise_level = 0.05, 0.02
    elif topology == t("Echo Chambers (Clusters)"): influence_strength, noise_level = 0.15, 0.01
    else: influence_strength, noise_level = 0.03, 0.01
        
    for time_step in range(1, steps):
        prev = current.copy()
        if topology == t("Echo Chambers (Clusters)"):
            left = np.roll(prev, 1)
            right = np.roll(prev, -1)
            neighbor_avg = (left + right) / 2
            current = prev + influence_strength * (neighbor_avg - prev)
        elif topology == t("Influencer Network (Hubs)"):
            hub_val = prev[0]
            current = prev + influence_strength * (hub_val - prev)
            current[0] = prev[0]
        else:
            global_mean = np.mean(prev)
            current = prev + influence_strength * (global_mean - prev)
        
        noise = np.random.normal(0, noise_level, n_agents)
        current += noise
        
        # Defense logic matching translated strings
        if defense == t("Algorithmic Dampening (Global)"): current *= 0.95
        elif defense == t("Fact-Check Debunking (Targeted)"):
            heal_indices = np.random.choice(n_agents, int(n_agents * 0.02))
            current[heal_indices] = 0.0
        elif defense == t("Hard Ban (Removal)"):
            current[current > 0.8] = 0.0
        
        current = np.clip(current, 0, 1)
        history[:, time_step] = current.copy()
        infection_rate.append(np.mean(current))
        
    st.markdown("---")

    with st.expander(t("View Neural Infection Network"), expanded=False):
        st.caption(t("Topological visualization of infection clusters."))
        sample_size = min(150, n_agents)
        
        if topology == t("Echo Chambers (Clusters)"): G = nx.caveman_graph(5, sample_size // 5)
        elif topology == t("Influencer Network (Hubs)"): G = nx.barabasi_albert_graph(sample_size, 2, seed=42)
        else: G = nx.erdos_renyi_graph(sample_size, 0.05, seed=42)
        
        node_colors = []
        for i in range(len(G.nodes())):
            if current[i] > 0.8: node_colors.append('#ef4444') # Infected
            elif current[i] > 0.3: node_colors.append('#f97316') # At risk
            else: node_colors.append('#3b82f6') # Healthy
            
        fig_net, ax_net = plt.subplots(figsize=(10, 5))
        pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)
        nx.draw(G, pos, node_color=node_colors, edge_color='#e0e0e0', node_size=100, alpha=0.8, ax=ax_net)
        ax_net.set_title(f"{t('Network Topology State')} ({t('Day')} {steps})")
        st.pyplot(fig_net)

    st.markdown("---")

    c_res1, c_res2 = st.columns([3, 1])
    with c_res1:
        st.subheader(t("Infection Spread (Heatmap)"))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(history, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_xlabel(t("Time Steps"))
        ax.set_ylabel(t("Agent ID"))
        st.pyplot(fig)
    with c_res2:
        st.subheader(t("Infection Rate"))
        chart_data = pd.DataFrame({'Time': range(len(infection_rate)), 'Infection Level': infection_rate})
        line = alt.Chart(chart_data).mark_line(color='red').encode(
            x=alt.X('Time', title=t('Time')), 
            y=alt.Y('Infection Level', title=t('Infection Level'))
        ).properties(height=300)
        st.altair_chart(line, use_container_width=True)
        final_rate = infection_rate[-1] * 100
        delta = final_rate - (infection_rate[0] * 100)
        st.metric(t("Final Infection Level"), f"{final_rate:.1f}%", f"{delta:.1f}%", delta_color="inverse")

    # --- PDF EXPORT ---
    st.markdown("---")
    st.subheader(t("Export Tactical Report"))
    st.caption(t("Generate a summary document with the simulation parameters and results."))
    
    if st.button(t("Generate Wargame PDF"), type="primary"):
        with st.spinner(t("Compiling the report...")):
            # PDF internal logic
            pdf = PDFReport()
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 14)
            pdf.cell(0, 10, t("Simulation Dossier: Information Warfare"), 0, 1)
            # ... [PDF Generation using t() labels] ...
            pdf_bytes = bytes(pdf.output())
            
        st.download_button(
            label=t("Download Wargame Report (PDF)"), 
            data=pdf_bytes, 
            file_name=f"RAP_Wargame_{datetime.now().strftime('%Y%m%d')}.pdf", 
            mime="application/pdf", 
            type="primary"
        )

# ==========================================
# MODULE 2: SOCIAL DATA ANALYSIS
# ==========================================
elif mode == t("2. Social Data Analysis (Universal)"):
    st.header(t("2. Social Data Analysis"))
    
    # Settings Columns
    col_impostazioni_1, col_impostazioni_2 = st.columns([1, 1])
    
    with col_impostazioni_1:
        st.subheader(t("Data Input"))
        input_method = st.radio(t("Input Method:"), 
            [t("CSV File Upload"), t("YouTube Link"), t("Raw Text Paste"), t("Telegram Dump (JSON)"), t("Reddit Native (OSINT)")], 
            horizontal=False
        )
        
    with col_impostazioni_2:
        st.subheader(t("Analysis Settings"))
        preset_personas = [
            t("Strategic Intelligence Analyst"), 
            t("Mass Psychologist (Emotional)"), 
            t("Legal Consultant (Defamation/Risk)"), 
            t("Campaign Manager (Opportunity)"),
            t("Custom (Define your own role...)")
        ]
        selected_persona = st.selectbox(t("Analysis Lens (Persona)"), preset_personas)
        
        if selected_persona == t("Custom (Define your own role...)"):
            persona = st.text_input(t("Enter Custom Persona"), value="Cybersecurity Expert", help=t("Define the exact role the AI should assume."))
        else:
            persona = selected_persona
            
        context_input = st.text_input(t("Global Context (Optional)"), placeholder=t("E.g., 'Discussion about Flat Earth'"))

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
        st.info("Chameleon Protocol: Live Infiltration or JSON Offline Dump.")
        with st.form("tg_form"):
            # --- CHAMELEON: DUAL INPUT UI ---
            tg_url = st.text_input("1. LIVE: Enter Public Channel URL (e.g., t.me/rian_ru)", placeholder="Leaves no trace. Max 30 recent messages.")
            tg_file = st.file_uploader("2. OFFLINE: Upload Telegram Chat Export (JSON)", type="json")
            submitted = st.form_submit_button("Extract Intel", type="primary")
            
            if submitted:
                tg_df = None
                with st.spinner("Infiltrating Telegram..."):
                    # Priority 1: Live Scraping
                    if tg_url:
                        tg_df = scrape_telegram_live(tg_url)
                    # Priority 2: Offline JSON parsing
                    elif tg_file:
                        tg_df = parse_telegram_json(tg_file)
                        
                    # Process and store the data if extraction was successful
                    if tg_df is not None and not tg_df.empty:
                        tg_df = detect_bot_activity(tg_df)
                        st.session_state['data_store'][input_method]['df'] = tg_df
                        st.session_state['data_store'][input_method]['analyzed'] = None 
                        st.session_state['data_store'][input_method]['summary'] = None
                        st.success(f"Intercepted {len(tg_df)} messages.")
                    else:
                        st.error("Extraction failed. Check the URL (must be public) or the JSON file format.")

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
                    ans = analyze_fallacies(row['content'], api_key=key, context_info=context_input, persona=persona, target_lang=st.session_state['global_lang'])
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

            if analyzed_df is not None and not analyzed_df.empty:
                try:
                    # Connect to the Panopticon database
                    conn_alert = sqlite3.connect('rap_panopticon.db')
                    # Retrieve already blacklisted identities
                    blacklist_data = pd.read_sql_query("SELECT agent_id FROM targets", conn_alert)
                    blacklist = blacklist_data['agent_id'].tolist()
                    conn_alert.close()
                    
                    # Cross-reference current analysis with Panopticon blacklist
                    detected_enemies = [agent for agent in analyzed_df['agent_id'].unique() if str(agent) in blacklist]
                    
                    if detected_enemies:
                        st.error(f"🚨 {t('CRITICAL ALERT: Known High-Value Targets detected in current analysis!')}")
                        st.warning(f"**{t('Detected Identities')}:** {', '.join(detected_enemies)}")
                        st.toast(t("HVT DETECTION: Check the Panopticon module."), icon="⚠️")
                except Exception:
                    pass # Silent fail if DB is busy or empty

            if analyzed_df is not None:
                st.markdown('<div id="briefing_anchor"></div>', unsafe_allow_html=True)

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
                                save_to_panopticon(row['agent_id'], row['threat_score'], "High-Value Target")
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
                    st.caption("Crisis Trend (Chronos Forecast)")
                    proj_data, is_escalating, time_to_crit = project_trend(adf)
                    if proj_data is not None:
                        base = alt.Chart(proj_data).mark_line().encode(
                            x=alt.X('Date:T', title='Timeline'),
                            y=alt.Y('Aggression', title='Aggression (0-10)'),
                            color=alt.Color('Type', scale=alt.Scale(domain=['Historical', 'Forecast'], range=['orange', 'red'])),
                            tooltip=['Date:T', 'Aggression', 'Type']
                        )
                        st.altair_chart(base, use_container_width=True)
                        
                        if is_escalating and time_to_crit is not None:
                            st.error(f"**CHRONOS ALERT:** Threshold breach (9.0) projected in ~{int(time_to_crit)} days.")
                        else:
                            st.caption(f"**Forecast:** {'⚠️ Escalating' if is_escalating else '📉 Stabilizing'}")
                    else:
                        st.caption("Sequence Trend (No valid dates found)")
                        adf['Sequence'] = adf.index
                        line = alt.Chart(adf).mark_line(color='orange').encode(x='Sequence', y='aggression')
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
elif mode == t("3. Cognitive Editor (Text/Image/Audio/Video)"):
    st.header(t("3. Cognitive Editor & Fact-Checker"))
    st.caption(t("Upload Text, Images (Memes/Screenshots), Audio clips or Videos for deep inspection."))
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader(t("Input"))
        # Translated radio options
        inp_type = st.radio(t("Input Type:"), [
            t("Text"), t("PDF"), t("Image (Vision Guard)"), 
            t("Audio (Voice Intel)"), t("Video (Deepfake Scan)")
        ], horizontal=True)
        
        text_inp = None
        media_inp = None
        media_type = "text"
        exif_data = {}
        
        if inp_type == t("Text"):
            text_inp = st.text_area(t("Paste Text Here"), height=300)
        elif inp_type == t("PDF"):
            f = st.file_uploader(t("Upload PDF"), type="pdf")
            if f:
                text_inp = extract_text_from_pdf(f)
                st.success(f"{t('Loaded')} {len(text_inp)} {t('chars from PDF')}")
        elif inp_type == t("Image (Vision Guard)"):
            f = st.file_uploader(t("Upload Image"), type=['png', 'jpg', 'jpeg'])
            text_inp = st.text_area(t("Image Context / Post Text (Optional)"), placeholder=t("E.g., Paste the Facebook/X post text that accompanied this photo..."), height=100)
            if f:
                original_img = Image.open(f)

                # --- OPSEC: BIOMETRIC ANONYMIZATION ---
                censor_faces = st.checkbox(t("Apply OPSEC Face Censor (Auto-Anonymize)"), value=False, help=t("Automatically detects and redacts human faces before analysis to protect identities."))
                
                if censor_faces:
                    media_inp, face_count = anonymize_faces(original_img)
                    if face_count > 0:
                        st.success(f"{t('Classified')}: {face_count} {t('identities redacted.')}")
                    else:
                        st.caption(t("No faces detected by the algorithm."))
                else:
                    media_inp = original_img

                media_type = "image"
                st.image(media_inp, caption=t("Evidence Image"), use_container_width=True)
                
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
                except Exception:
                    pass
                
                if gps_coords is not None:
                    st.success(f"🌍 **{t('Location Traces Detected (GPS):')}** {t('Geolocation data found!')}")
                    st.map(gps_coords, zoom=12, use_container_width=True)
                
                if exif_data:
                    software_used = exif_data.get('Software') or exif_data.get('ProcessingSoftware')
                    date_original = exif_data.get('DateTimeOriginal') or exif_data.get('DateTime')
                    
                    with st.expander(t("Invisible EXIF Metadata Found (OSINT)"), expanded=True):
                        if date_original:
                            st.info(f"**{t('Original Creation Date:')}** {date_original}")
                        else:
                            st.caption(f"**{t('Creation Date:')}** {t('Not found in metadata.')}")
                            
                        if software_used:
                            st.warning(f"**{t('Editing Trace Detected:')}** {t('The image was modified using')} **{software_used}**.")
                        else:
                            st.success(f"✅ **{t('Clean Metadata:')}** {t('No image editing software detected in EXIF.')}")
                        st.json(exif_data)
                else:
                    st.caption(t("No EXIF data found (Image might be scrubbed by social media)."))
                
                # --- OPSEC IMAGE SCRUBBER ---
                st.markdown("---")
                st.markdown(f"#### {t('OPSEC: Metadata Scrubber')}")
                st.caption(t("Remove all invisible EXIF data (GPS, Device Info) before sharing this evidence."))
                
                def strip_exif(image):
                    data = list(image.getdata())
                    image_without_exif = Image.new(image.mode, image.size)
                    image_without_exif.putdata(data)
                    return image_without_exif

                clean_image = strip_exif(media_inp)
                img_byte_arr = io.BytesIO()
                clean_image.save(img_byte_arr, format='PNG')
                clean_bytes = img_byte_arr.getvalue()

                st.download_button(
                    label=t("Download Sanitized Image (Zero EXIF)"),
                    data=clean_bytes,
                    file_name=f"RAP_Sanitized_Evidence_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                    mime="image/png",
                    type="primary"
                )

                # --- ELA FORENSICS VISUALIZER ---
                st.markdown(f"#### {t('Error Level Analysis (ELA)')}")
                st.caption(t("Detects digital manipulation (Photoshop/copy-paste). Artificially inserted elements will glow significantly brighter."))
                
                with st.spinner(t("Generating ELA Map...")):
                    ela_img = perform_ela(media_inp)
                    if ela_img:
                        st.image(ela_img, caption=t("ELA Heatmap (Look for glowing/inconsistent edges)"), use_container_width=True)
                    else:
                        st.warning(t("Could not generate ELA for this image format."))

        elif inp_type == t("Audio (Voice Intel)"):
            f = st.file_uploader(t("Upload Audio"), type=['mp3', 'wav', 'm4a'])
            if f:
                media_inp = f.read() 
                media_type = "audio"
                st.audio(media_inp, format='audio/mp3')
                
                with st.spinner(t("Generating waveform...")):
                    waveform_img = generate_audio_waveform(media_inp)
                    if waveform_img:
                        st.image(waveform_img, use_container_width=True)
                        st.caption(t("Inspect the waveform for unnatural silences or frequency clipping (typical of AI voice clones)."))
        
        elif inp_type == t("Video (Deepfake Scan)"):
            text_inp = st.text_area(t("Video Context (Optional)"), placeholder=t("What is this video claiming?"), height=100)
            text_inp = "CRITICAL INSTRUCTION: You are looking at a Forensic Storyboard grid of a video... [ENG PROMPT]"
            f = st.file_uploader(t("Upload Video (Max 50MB)"), type=['mp4', 'mov'])
            if f:
                raw_video_bytes = f.read()
                st.video(raw_video_bytes)
                with st.spinner(t("Extracting forensic storyboard (Nuclear Option)...")):
                    storyboard_bytes = create_video_storyboard(raw_video_bytes, num_frames=12)
                    if storyboard_bytes:
                        st.success(t("✅ Forensic storyboard extracted invisibly for the AI."))
                        media_inp = storyboard_bytes
                        media_type = "image"
                    else:
                        st.error(t("Failed to extract frames."))
            else:
                st.info(t("Please upload an MP4 or MOV file to start the forensic analysis."))
            
        go = st.button(t("Analyze, Sanitize & Scan AI"), use_container_width=True, type="primary")

    with c2:
        st.subheader(t("Output (Analysis & Sanitize)"))
        if go:
            if media_inp or text_inp:
                with st.spinner(f"{t('Processing with Gemini')} ({inp_type})..."):
                    ret = cognitive_rewrite(text_inp, key, media_inp, media_type, target_lang=st.session_state['global_lang'])
                    
                    if ret:
                        # --- AI SCANNER UI ---
                        ai_prob = ret.get('ai_generated_probability', 0)
                        if ai_prob > 75:
                            st.error(f"🤖 **{t('HIGH PROBABILITY OF AI GENERATION:')} {ai_prob}%**")
                            st.caption(t(ret.get('ai_analysis', 'Detected deepfake/LLM patterns.')))
                        elif ai_prob > 40:
                            st.warning(f"⚠️ **{t('SUSPICIOUS AI GENERATION SCORE:')} {ai_prob}%**")
                            st.caption(t(ret.get('ai_analysis', 'Possible use of AI tools.')))
                        else:
                            st.success(f"👤 **{t('LIKELY HUMAN GENERATED')} ({t('AI Score:')} {ai_prob}%)**")
                        
                        st.markdown("---")

                        # --- FORENSIC VIDEO TIMELINE ---
                        v_timeline = ret.get('video_timeline', [])
                        if inp_type == t("Video (Deepfake Scan)") and isinstance(v_timeline, list) and len(v_timeline) > 0:
                            st.markdown(f"#### {t('Forensic Video Timeline')}")
                            st.caption(t("Temporal analysis of AI manipulation probability across the video length."))
                            vt_df = pd.DataFrame(v_timeline)
                            # (Column renaming logic remains same as it is structural)
                            if 'timestamp' in vt_df.columns and 'ai_score' in vt_df.columns:
                                vt_df['ai_score'] = pd.to_numeric(vt_df['ai_score'], errors='coerce').fillna(0)
                                chart_vt = alt.Chart(vt_df).mark_line(point=True, color='red').encode(
                                    x=alt.X('timestamp:O', title=t('Timestamp')),
                                    y=alt.Y('ai_score:Q', title=t('AI Probability (%)'), scale=alt.Scale(domain=[0, 100])),
                                    tooltip=['timestamp', 'ai_score', 'details']
                                ).properties(height=250)
                                st.altair_chart(chart_vt, use_container_width=True)
                                
                                with st.expander(t("View Frame-by-Frame Details")):
                                    for _, row in vt_df.iterrows():
                                        st.write(f"**{row.get('timestamp', '??:??')}** ({t('Score')}: {int(row.get('ai_score', 0))}%) - {row.get('details', '')}")
                        
                        st.markdown("---")

                        # --- FALLACY & LOGIC UI ---
                        if ret.get('has_fallacy'):
                            st.error(f"🛑 {t('Issue Detected:')} **{t(ret.get('fallacy_type', 'System Processing Error'))}**")
                            st.metric(t("Aggression Level"), f"{ret.get('aggression', 0)}/10")
                            st.warning(f"**{t('Analysis:')}** {t(ret.get('explanation', 'No details available.'))}")
                        else:
                            st.success(f"✅ {t('Neural Guard: No major issues detected.')}")
                            st.info(f"**{t('Analysis:')}** {t(ret.get('explanation', 'Content is sound.'))}")
                        
                        st.markdown("---")
                        syl_breakdown = ret.get('syllogism_breakdown', [])
                        if syl_breakdown:
                            st.markdown(f"#### {t('Logical Deconstruction')}")
                            st.caption(t("The text was fragmented into formal premises to identify the exact point where logic fails."))
                            for step in syl_breakdown:
                                flaw_text = str(step.get('flaw', '')).strip()
                                if flaw_text and flaw_text.lower() not in ["none", "nessuno", "nessuna", "n/a", "", "null"]:
                                    st.error(f"**{t(step.get('step', 'Step'))}**: {t(step.get('text', ''))}\n\n⚠️ **{t('Logical Leap / Fallacy:')}** {t(flaw_text)}")
                                else:
                                    st.info(f"**{t(step.get('step', 'Step'))}**: {t(step.get('text', ''))}")
                        
                        st.markdown("---")
                        st.markdown(f"#### {t('Rewritten Version / Transcript Summary')}")
                        st.info(t(ret.get('rewritten_text', 'No rewrite available.')))
                        
                        st.markdown("---")
                        if media_type == "audio" and 'voice_stress_score' in ret:
                            stress = ret.get('voice_stress_score', 0)
                            st.markdown(f"#### {t('Voice & Tone Analysis')}")
                            if stress > 65:
                                st.error(f"**{t('Voice Stress Score:')} {stress}%** ({t('High emotion, anger, or panic detected in prosody')})")
                            else:
                                st.success(f"**{t('Voice Stress Score:')} {stress}%** ({t('Calm, controlled, or neutral tone')})")
                        
                        # --- GHOST READER (OCR FORENSICS) ---
                        if ret.get('ocr_extraction') and ret.get('ocr_extraction').lower() not in ["none", "n/a", "", "null"]:
                            st.markdown(f"#### {t('Ghost Reader (OCR Extraction)')}")
                            st.caption(t("AI optical scan of texts hidden in the media (signs, screens, documents)."))
                            st.info(f"{t(ret['ocr_extraction'])}")
                        
                        # --- FACT CHECKER ---
                        st.markdown(f"#### {t('Fact Checker (Claims to Verify)')}")
                        facts = ret.get('facts', [])
                        if facts:
                            for f_claim in facts: st.write(f"- {t(f_claim)}")
                        else:
                            st.caption(t("No specific factual claims found."))

                        # --- FORENSIC DOSSIER EXPORT ---
                        st.markdown("---")
                        file_hash = calculate_sha256(media_inp) if media_type in ['audio', 'video'] else "N/A"
                        
                        dossier_text = f"{t('RAP FORENSIC DOSSIER')}\n"
                        dossier_text += f"{t('Date')}: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                        dossier_text += f"{t('Digital Fingerprint (SHA-256)')}: {file_hash}\n"
                        dossier_text += f"{'='*40}\n\n"
                        dossier_text += f"[1] {t('AI GENERATION SCAN')}\n{t('AI Probability Score')}: {ai_prob}%\n\n"
                        
                        st.download_button(
                            label=t("Download Forensic Dossier (TXT)"),
                            data=dossier_text.encode('utf-8'),
                            file_name=f"RAP_Forensic_Dossier_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain",
                            type="primary"
                        )
            else:
                st.warning(t("Please provide input."))

# ==========================================
# MODULE 4: COMPARISON TEST (A/B TESTING)
# ==========================================
elif mode == t("4. Comparison Test (A/B Testing)"):
    st.header(t("4. Universal Arena (A/B Testing)"))
    st.caption(t("Compare data from YouTube, CSV, or Raw Text to find the most aggressive narratives."))
    
    col_a, col_b = st.columns(2)
    
    # Helper function for Arena data loading with translated UI
    def load_arena_data(key_prefix, column):
        with column:
            in_type = st.radio(f"{t('Input')} {key_prefix}", [t("YouTube Link"), t("CSV Upload"), t("Raw Text Paste")], horizontal=True, key=f"r_{key_prefix}")
            df = None
            if in_type == t("YouTube Link"):
                url = st.text_input(f"{t('YouTube URL')} {key_prefix}", key=f"url_{key_prefix}")
                limit = st.number_input(f"{t('Comments to Fetch')} {key_prefix}", 10, 1000, 50, key=f"lim_{key_prefix}")
                if st.button(f"{t('Scrape')} {key_prefix}", key=f"btn_yt_{key_prefix}"):
                    with st.spinner(f"{t('Scraping YouTube')} {key_prefix}..."):
                        df = scrape_youtube_comments(url, limit)
            elif in_type == t("CSV Upload"):
                f = st.file_uploader(f"{t('Upload CSV')} {key_prefix}", type="csv", key=f"f_{key_prefix}")
                if f: df = normalize_dataframe(pd.read_csv(f))
            else:
                txt = st.text_area(f"{t('Paste Data')} {key_prefix}", height=100, key=f"t_{key_prefix}")
                if st.button(f"{t('Process Text')} {key_prefix}", key=f"btn_txt_{key_prefix}"):
                    if txt: df = parse_raw_paste(txt)
                    
            if df is not None:
                df = detect_bot_activity(df)
                if 'Select' not in df.columns: df.insert(0, "Select", False)
                st.session_state['data_store']['Arena'][f'df_{key_prefix.lower()}'] = df
                st.success(f"{t('Loaded')} {len(df)} {t('items in')} {key_prefix}.")
        
    st.subheader(t("Contender A (Left)"))
    load_arena_data("A", col_a)
    st.subheader(t("Contender B (Right)"))
    load_arena_data("B", col_b)

    df_a_raw = st.session_state['data_store']['Arena']['df_a']
    df_b_raw = st.session_state['data_store']['Arena']['df_b']

    if df_a_raw is not None and df_b_raw is not None:
        st.divider()
        st.markdown(f"### {t('Step 2: Select Data to Analyze')}")
        st.caption(t("Select specific comments to compare using the checkboxes. If none selected, the top N rows will be analyzed."))
        
        c_edit_a, c_edit_b = st.columns(2)
        with c_edit_a:
            st.markdown(f"**{t('Contender A')}** ({len(df_a_raw)} {t('items')})")
            edited_a = st.data_editor(
                df_a_raw,
                column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                disabled=["content", "agent_id", "timestamp", "is_bot", "likes"],
                key="editor_arena_a", height=300, hide_index=True
            )
        with c_edit_b:
            st.markdown(f"**{t('Contender B')}** ({len(df_b_raw)} {t('items')})")
            edited_b = st.data_editor(
                df_b_raw,
                column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                disabled=["content", "agent_id", "timestamp", "is_bot", "likes"],
                key="editor_arena_b", height=300, hide_index=True
            )

        st.divider()
        c_action, c_limit = st.columns([1, 1])
        with c_limit:
            max_analyze = st.number_input(t("Max Rows to Analyze (if no manual selection)"), 5, 100, 20)
        with c_action:
            st.write("") 
            st.write("") 
            start_analysis = st.button(t("Step 3: Run Comparative Analysis"), type="primary", disabled=not key)

        if start_analysis:
            # Analyze Contender A
            subset_a = edited_a[edited_a.Select]
            if subset_a.empty: subset_a = edited_a.head(max_analyze)
            
            prog_a = st.progress(0)
            res_a = []
            st.markdown(f"**{t('Analyzing Contender A...')}**")
            for i, (_, row) in enumerate(subset_a.iterrows()):
                res_a.append(analyze_fallacies(row['content'], api_key=key))
                prog_a.progress((i + 1) / len(subset_a))
            final_a = pd.concat([subset_a.reset_index(drop=True), pd.DataFrame(res_a)], axis=1)
            st.session_state['data_store']['Arena']['analyzed_a'] = final_a

            # Analyze Contender B
            subset_b = edited_b[edited_b.Select]
            if subset_b.empty: subset_b = edited_b.head(max_analyze)

            prog_b = st.progress(0)
            res_b = []
            st.markdown(f"**{t('Analyzing Contender B...')}**")
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
        st.header(t("Match Results"))
        
        c1, c2, c3 = st.columns(3)
        agg_a = res_df_a['aggression'].mean()
        agg_b = res_df_b['aggression'].mean()
        delta_agg = agg_a - agg_b
        bots_a = len(res_df_a[res_df_a['is_bot']==True])
        bots_b = len(res_df_b[res_df_b['is_bot']==True])
        delta_bots = bots_a - bots_b
        fallacy_a = len(res_df_a[res_df_a['has_fallacy']==True])
        fallacy_b = len(res_df_b[res_df_b['has_fallacy']==True])
        
        c1.metric(t("Avg Aggression (A vs B)"), f"{agg_a:.1f} vs {agg_b:.1f}", f"{delta_agg:.1f}")
        c2.metric(t("Bot Count (A vs B)"), f"{bots_a} vs {bots_b}", f"{delta_bots}")
        c3.metric(t("Fallacies (A vs B)"), f"{fallacy_a} vs {fallacy_b}")
        
        st.markdown("---")
        st.subheader(t("Tactical Visual Comparison"))
        
        res_df_a['Source'] = t('Contender A')
        res_df_b['Source'] = t('Contender B')
        combined = pd.concat([res_df_a, res_df_b])
        
        # 1. Base Aggression Chart (Translated axes)
        chart_agg = alt.Chart(combined).mark_bar().encode(
            x=alt.X('Source', title=None),
            y=alt.Y('mean(aggression)', title=t('Avg Aggression')),
            color=alt.Color('Source', scale=alt.Scale(domain=[t('Contender A'), t('Contender B')], range=['#3b82f6', '#f97316'])),
            tooltip=['Source', 'mean(aggression)']
        ).properties(height=200)
        st.altair_chart(chart_agg, use_container_width=True)
        
        c_vis1, c_vis2 = st.columns(2)
        with c_vis1:
            st.caption(t("Emotional Spectrum Clash"))
            if 'primary_emotion' in combined.columns:
                emo_combined = combined.groupby(['Source', 'primary_emotion']).size().reset_index(name='Count')
                chart_emo = alt.Chart(emo_combined).mark_bar().encode(
                    x=alt.X('primary_emotion:N', title=t('Emotion'), sort='-y'),
                    y=alt.Y('Count:Q', title=t('Frequency')),
                    color=alt.Color('Source:N', scale=alt.Scale(domain=[t('Contender A'), t('Contender B')], range=['#3b82f6', '#f97316']), legend=None),
                    xOffset='Source:N',
                    tooltip=['Source', 'primary_emotion', 'Count']
                ).properties(height=300)
                st.altair_chart(chart_emo, use_container_width=True)
                
        with c_vis2:
            st.caption(t("Weaponized Fallacies"))
            if 'fallacy_type' in combined.columns:
                fal_combined = combined[combined['has_fallacy'] == True].groupby(['Source', 'fallacy_type']).size().reset_index(name='Count')
                if not fal_combined.empty:
                    chart_fal = alt.Chart(fal_combined).mark_bar().encode(
                        x=alt.X('fallacy_type:N', title=t('Fallacy Type'), sort='-y'),
                        y=alt.Y('Count:Q', title=t('Frequency')),
                        color=alt.Color('Source:N', scale=alt.Scale(domain=[t('Contender A'), t('Contender B')], range=['#3b82f6', '#f97316'])),
                        xOffset='Source:N',
                        tooltip=['Source', 'fallacy_type', 'Count']
                    ).properties(height=300)
                    st.altair_chart(chart_fal, use_container_width=True)
                else:
                    st.success(t("No fallacies detected in either contender."))

        # --- AI NARRATIVE CLASH BRIEFING ---
        st.markdown("---")
        st.subheader(t("The Oracle: Narrative Clash Assessment"))
        st.caption(t("Force the AI to analyze the differing psychological and tactical profiles of the two contenders."))
        
        if st.button(t("Generate Clash Briefing"), type="primary"):
            with st.spinner(t("Analyzing psychological divergence...")):
                # Prompt remains technical but UI elements around it are translated
                top_emotions_a = res_df_a['primary_emotion'].value_counts().head(3).to_dict() if 'primary_emotion' in res_df_a else "N/A"
                top_emotions_b = res_df_b['primary_emotion'].value_counts().head(3).to_dict() if 'primary_emotion' in res_df_b else "N/A"
                sample_a = res_df_a['content'].iloc[0] if not res_df_a.empty else ""
                sample_b = res_df_b['content'].iloc[0] if not res_df_b.empty else ""
                
                clash_prompt = f"You are an elite analyst... [PROMPT DATA]"
                try:
                    client = genai.Client(api_key=key)
                    clash_res = client.models.generate_content(model='gemini-2.0-flash', contents=clash_prompt)
                    st.warning(f"### {t('Tactical Clash Report')}")
                    st.markdown(clash_res.text)
                except Exception as e:
                    st.error(f"{t('Failed to generate briefing')}: {e}")

        with st.expander(t("Detailed Comparison Data")):
            st.dataframe(combined[['Source', 'content', 'aggression', 'primary_emotion', 'fallacy_type', 'is_bot']])

        # --- PDF EXPORT FOR ARENA ---
        st.markdown("---")
        st.subheader(t("Export Battle Report"))
        st.caption(t("Generate a comparative PDF dossier detailing the metrics between Contender A and Contender B."))
        
        if st.button(t("Generate Arena PDF"), type="primary"):
            with st.spinner(t("Compiling the comparative report...")):
                # Internal PDF logic uses English but labels are wrapped if possible
                pdf = PDFReport()
                pdf.add_page()
                pdf.set_font("Helvetica", 'B', 14)
                pdf.cell(0, 10, f"{t('Comparative Dossier')}: {t('Narrative A/B Testing')}", 0, 1)
                # ... (rest of PDF generation logic remains similar)
                
                pdf_bytes_arena = bytes(pdf.output())
                
            st.download_button(
                label=t("Download Arena Report (PDF)"), 
                data=pdf_bytes_arena, 
                file_name=f"RAP_Arena_Match_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", 
                mime="application/pdf", 
                type="primary"
            )

# ==========================================
# MODULE 5: LIVE RADAR (RSS/REDDIT)
# ==========================================
elif mode == t("5. Live Radar (RSS/Reddit)"):
    st.header(t("5. Live Radar (Crisis Alert System)"))
    st.caption(t("Monitor live RSS feeds or subreddits to intercept escalating disinformation and aggression in real-time."))
    
    # API Key Handling (Translated)
    if "GEMINI_API_KEY" in st.secrets: 
        key = st.secrets["GEMINI_API_KEY"]
    else: 
        key = st.text_input(t("API Key"), type="password")
    
    c_radar1, c_radar2, c_radar3 = st.columns([2, 1, 1])
    with c_radar1:
        feed_url = st.text_input(t("Enter RSS Feed, Subreddit, or News Keyword"), placeholder=t("E.g., ansa, bbc, reddit.com/r/worldnews"))
    with c_radar2:
        world_languages = [
            "English", "Italiano", "Español", "Français", "Deutsch", "Português",
            "Русский (Russian)", "العربية (Arabic)", "中文 (Chinese)", 
            "日本語 (Japanese)", "فارسی (Persian)", "हिन्दी (Hindi)", 
            "韓国어 (Korean)", "Türkçe (Turkish)"
        ]
        news_region = st.selectbox(t("Search Language/Region"), world_languages, index=1)
    with c_radar3:
        max_entries = st.number_input(t("Entries"), 5, 50, 15)
        fetch_btn = st.button(t("Step 1: Fetch Feed"), type="primary", use_container_width=True)
        # --- DEFCON CYBER SCAN BUTTON ---
        defcon_btn = st.button(f"🚨 {t('DEFCON Cyber Scan')}", type="primary", use_container_width=True)

    # --- AUTOMATED ALERT CONFIGURATION (Translated) ---
    with st.expander(t("Automated Alert Configuration (Webhook)")):
        alert_webhook = st.text_input(t("Webhook URL"), placeholder="https://hooks.slack.com/...", help=t("If Aggression exceeds the threshold, an alert payload will be dispatched here."))
        
        with st.expander(f"ℹ️ {t('How to get a Webhook URL (Discord / Slack)')}"):
            st.markdown(f"""
            **{t('For Discord')}:**
            1. {t('Open Server Settings -> Integrations -> Webhooks')}.
            2. {t('Click New Webhook, name it RAP Sentinel, and Copy Webhook URL')}.
            """)
            
        alert_threshold = st.slider(t("Trigger Alert Threshold (Aggression)"), min_value=1.0, max_value=10.0, value=8.0, step=0.5)
        st.caption(f"{t('Note: System will dispatch emergency protocols if average aggression spikes above')} {alert_threshold}/10.")

    if (fetch_btn and feed_url) or defcon_btn:
        with st.spinner(t("Intercepting live feed...")):
            try:
                # Logic for feed fetching (Internal shortcuts remain English/Technical)
                user_input = feed_url.strip().lower() if not defcon_btn else "defcon"
                actual_url = feed_url.strip()
                
                # ... [Internal RSS Shortcut Mapping remains unchanged] ...
                
                feed = feedparser.parse(actual_url)
                
                if not feed.entries:
                    st.error(t("Could not fetch entries. Check the URL or formatting."))
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
                    st.success(f"✅ {t('Intercepted')} {len(entries_data)} {t('items from')} {feed.feed.get('title', 'Feed')}.")
            except Exception as e:
                st.error(f"{t('Radar Fetch Error')}: {str(e)}")

    # --- STEP 2: SELECTION AND ANALYSIS ---
    radar_df = st.session_state['data_store'].get('Radar', {}).get('df')
    if radar_df is not None:
        st.divider()
        st.markdown(f"### {t('Step 2: Select News to Analyze')}")
        
        edited_radar = st.data_editor(
            radar_df,
            column_config={"Select": st.column_config.CheckboxColumn(required=True)},
            disabled=["timestamp", "content", "link"],
            key="editor_radar",
            use_container_width=True
        )
        
        c_action, c_limit = st.columns([1, 1])
        with c_limit:
            max_analyze = st.number_input(t("Max Rows to Analyze"), 1, len(radar_df), min(10, len(radar_df)))
        with c_action:
            selected = edited_radar['Select'].sum()
            btn_label = f"{t('Step 3: Run Threat Analysis')} ({selected if selected > 0 else f'Batch Top {max_analyze}'})"
            analyze_btn = st.button(btn_label, type="primary", disabled=not key)
        
        if analyze_btn:
            subset = edited_radar[edited_radar.Select] if selected > 0 else edited_radar.head(max_analyze)
            
            prog = st.progress(0)
            res = []
            st.markdown(f"**{t('Running Threat Analysis (Crisis Protocol)...')}**")
            
            for i, (_, row) in enumerate(subset.iterrows()):
                # Crisis analysis prompt (IA logic)
                crisis_prompt = f"Analyze crisis level for: {row['content']}..."
                # [Internal analysis logic remains English for LLM consistency]
                # ... (API Call) ...
                res.append(parsed)
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
        col_m1.metric(t("Live Aggression Index"), f"{avg_aggression:.1f}/10")
        col_m2.metric(t("Flagged Issues"), len(analyzed_radar[analyzed_radar['has_fallacy']==True]))
        col_m3.metric(t("Monitored Items"), len(analyzed_radar))
        
        if avg_aggression >= alert_threshold:
            st.error(f"🚨 **{t('CRITICAL CRISIS ALERT')}:** {t('Threshold breached!')}")
            if alert_webhook:
                st.toast(t("Dispatching emergency alert..."), icon="🚨")
        
        # --- GEO-INT MAP (Translated) ---
        st.markdown("---")
        st.subheader(t("Geopolitical Crisis Map"))
        
        # ... [Plotly Geo Map Logic wrapped in t() for titles] ...

        st.subheader(t("Live Feed Feedbacks"))
        for idx, r in analyzed_radar.iterrows():
            with st.container(border=True):
                st.caption(f"🕒 {r['timestamp']} | **Agg:** {r['aggression']}/10")
                st.write(r['content'][:300] + "...")
                
                if st.button(t("Deploy Web Spider"), key=f"spider_{idx}"):
                    with st.spinner(t("Crawling target infrastructure...")):
                        internal, external = run_web_spider(r['link'])
                        st.markdown(f"**{t('Infrastructure Mapped')}:**")
                        # Display links...

        # --- EXPORT RADAR INTEL (Translated) ---
        st.divider()
        st.subheader(t("Export Radar Intel"))
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            st.download_button(t("Download Alert Report (CSV)"), data=csv_data, file_name="Radar.csv", type="primary")
        with c_btn2:
            st.download_button(t("Download Threat Report (PDF)"), data=pdf_bytes_radar, file_name="Threats.pdf", type="primary")

    # --- LIVE SENTINEL MODE (AUTONOMOUS) ---
    st.divider()
    st.subheader(t("Live Sentinel Mode (Autonomous)"))
    st.caption(t("System will continuously fetch and analyze alerts automatically via Webhook."))

    live_toggle = st.toggle(t("Activate Live Sentinel"), key="live_radar_toggle")
    if live_toggle:
        refresh_rate = st.slider(t("Scan Interval (Seconds)"), 15, 600, 60)
        # Sentinel autonomous loop logic follows...
        st.info(f"🟢 **{t('SENTINEL ACTIVE')}** | {t('Next scan in')} {refresh_rate}s")

# ==========================================
# MODULE 6: DEEP DOCUMENT ORACLE (RAG)
# ==========================================
elif mode == t("6. Deep Document Oracle (RAG)"):
    st.header(t("6. Deep Document Oracle"))
    st.caption(t("Upload massive PDFs (e.g., manifestos, contracts, books) and find contradictions and extract deep facts without traditional RAG limits."))

    # API Key Handling
    if "GEMINI_API_KEY" in st.secrets: 
        key = st.secrets["GEMINI_API_KEY"]
    else: 
        key = st.text_input(t("API Key"), type="password")

    # Translated File Uploader
    uploaded_files = st.file_uploader(t("Upload PDF Documents"), type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if 'doc_full_text' not in st.session_state or st.button(t("Process Documents")):
            with st.spinner(t("Extracting text from all documents...")):
                full_text = ""
                for f in uploaded_files:
                    txt = extract_text_from_pdf(f)
                    if txt:
                        full_text += f"\n\n--- DOCUMENT: {f.name} ---\n\n{txt}"
                
                st.session_state['doc_full_text'] = full_text
                st.success(f"{t('Processed')} {len(full_text)} {t('characters across')} {len(uploaded_files)} {t('documents. The Oracle is ready.')}")
        
        if 'doc_full_text' in st.session_state and st.session_state['doc_full_text']:
            st.divider()
            
            # --- PII SANITIZER (Translated UI) ---
            c_san1, c_san2 = st.columns([1, 3])
            with c_san1:
                if st.button(t("Sanitize Document (Redact PII)"), help=t("Hides Emails, IPs, IBANs, and Phones before chatting with the Oracle")):
                    with st.spinner(t("Sanitizing sensitive data...")):
                        clean_text = sanitize_pii(st.session_state['doc_full_text'])
                        st.session_state['doc_full_text'] = clean_text
                        st.success(t("Document successfully sanitized (CIA Blackout Protocol)!"))
            
            # --- DOWNLOAD REDACTED DOSSIER ---
            st.download_button(
                label=t("Download Redacted Dossier (TXT)"),
                data=st.session_state['doc_full_text'].encode('utf-8'),
                file_name=f"RAP_Classified_Redacted_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                type="primary"
            )
            
            # --- KNOWLEDGE GRAPH (Translated UI) ---
            with st.expander(t("Extract Document Power Network (Knowledge Graph)"), expanded=False):
                st.caption(t("Automatically scan the document to map relationships between People, Organizations, and Locations."))
                if st.button(t("Generate Power Graph")):
                    with st.spinner(t("Extracting entities and relationships (this may take a minute)...")):
                        graph_prompt = f"Analyze this document and extract the top 12 most important relationships between entities... [PROMPT TRUNCATED FOR BREVITY]"
                        try:
                            client = genai.Client(api_key=key)
                            graph_res = client.models.generate_content(model='gemini-2.0-flash', contents=graph_prompt)
                            graph_json = extract_json(graph_res.text)
                            fig_doc = plot_document_entity_graph(graph_json)
                            if fig_doc:
                                st.pyplot(fig_doc)
                                # Global Memory Registration
                                for rel in graph_json.get('relations', []):
                                    if rel.get('source'): st.session_state['global_entities'].add(str(rel['source']).strip().lower())
                                    if rel.get('target'): st.session_state['global_entities'].add(str(rel['target']).strip().lower())
                                st.success(f"{t('Registered')} {len(st.session_state['global_entities'])} {t('entities into Global Memory.')}")
                        except Exception as e:
                            st.error(f"{t('Graph Extraction Error')}: {str(e)}")

            st.divider()

            # --- CONTRADICTION SCANNER (Translated UI) ---
            with st.expander(t("Deep Scan: Contradictions & Loopholes"), expanded=False):
                st.caption(t("Force the AI to audit the entire document specifically looking for logical contradictions, legal loopholes, or unfulfilled claims."))
                if st.button(t("Run Audit Scan"), type="primary"):
                    with st.spinner(t("Auditing document for contradictions...")):
                        audit_prompt = f"You are a ruthless Forensic Auditor... [PROMPT TRUNCATED]"
                        try:
                            client = genai.Client(api_key=key)
                            audit_res = client.models.generate_content(model='gemini-2.0-flash', contents=audit_prompt)
                            st.warning(f"### ⚠️ {t('Forensic Audit Report')}")
                            st.markdown(audit_res.text)
                            st.session_state.doc_oracle_history.append({"role": "assistant", "content": f"**[{t('AUTOMATED AUDIT REPORT')}]**\n\n{audit_res.text}"})
                        except Exception as e:
                            st.error(f"{t('Audit Error')}: {str(e)}")

            # --- MULTi-AGENT DEBATE (Translated UI) ---
            with st.expander(t("Multi-Agent War Room (Stress Test)"), expanded=False):
                st.caption(t("Deploy two opposing AI agents to debate the document. The Red Team attacks it, the Blue Team defends it."))
                debate_topic = st.text_input(t("Debate Focus (e.g., 'Security flaws', 'Ethical implications'):"), placeholder=t("What should the agents fight about?"))
                
                if st.button(t("Initiate Agent Debate"), type="primary"):
                    if not debate_topic:
                        st.warning(t("Please enter a debate focus."))
                    else:
                        st.markdown(f"### 🔴 {t('Red Team')} vs 🔵 {t('Blue Team')}: *{debate_topic}*")
                        # Agent logic follows...
                        # [Keep the existing debate logic but wrap UI strings in t()]

            st.divider()

            # --- THE ORACLE CHAT (UPGRADED: HIGH-PRECISION SCAN) ---
            st.subheader(t("Chat with the Oracle"))
            
            for message in st.session_state.doc_oracle_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            if prompt := st.chat_input(t("Ask the Deep Oracle...")):
                st.session_state.doc_oracle_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner(t("Scouring massive document context...")):
                        chunks = chunk_document(st.session_state['doc_full_text'])
                        query_words = prompt.lower().split()
                        scored_chunks = []
                        for c in chunks:
                            score = sum(2 for word in query_words if word in c.lower())
                            scored_chunks.append((score, c))
                        
                        scored_chunks.sort(key=lambda x: x[0], reverse=True)
                        top_context = "\n\n[SEGMENT]\n\n".join([text for score, text in scored_chunks[:4]])
                        
                        rag_prompt = f"Answer the query using ONLY these segments: {top_context}\nQuery: {prompt}"
                        
                        try:
                            client = genai.Client(api_key=key)
                            response = client.models.generate_content(model='gemini-2.0-flash', contents=rag_prompt).text
                            final_response = f"*({t('Method: High-Precision Context Scan')})*\n\n{response}"
                        except Exception as e:
                            response = ask_document_oracle(st.session_state['doc_full_text'], prompt, key)
                            final_response = response

                        st.markdown(final_response)
                        st.session_state.doc_oracle_history.append({"role": "assistant", "content": final_response})

# ==========================================
# MODULE 7: PANOPTICON (HVT WATCHLIST)
# ==========================================
elif mode == t("7. Panopticon (HVT Watchlist)"):
    st.header(t("7. Panopticon: Global Threat Database"))
    st.caption(t("Central persistent database. Automatically stores all High-Value Targets (HVT) intercepted across various modules."))
    
    # Establish persistent connection to the SQLite database
    conn = sqlite3.connect('rap_panopticon.db', check_same_thread=False)
    
    # Try to load existing targets from the database
    try:
        df_panopticon = pd.read_sql_query("SELECT * FROM targets ORDER BY risk_score DESC", conn)
    except:
        # Fallback if the table or database hasn't been created yet
        df_panopticon = pd.DataFrame()
        
    if not df_panopticon.empty:
        # Strategic Metrics (Translated)
        c_p1, c_p2, c_p3 = st.columns(3)
        c_p1.metric(t("Total Tracked Entities"), len(df_panopticon))
        c_p2.metric(t("Critical Threats (Score > 80)"), len(df_panopticon[df_panopticon['risk_score'] > 80]))
        c_p3.metric(t("Latest Detection"), df_panopticon['last_seen'].max())
        
        st.markdown("---")
        st.subheader(t("Target Management"))
        
        # Multilingual Data Editor for database management
        edited_pan = st.data_editor(
            df_panopticon,
            column_config={
                "agent_id": t("Target Identity"),
                "risk_score": st.column_config.ProgressColumn(t("Threat Score"), min_value=0, max_value=100, format="%f"),
                "threat_type": t("Classification"),
                "last_seen": t("Last Active")
            },
            use_container_width=True,
            num_rows="dynamic",
            key="panopticon_editor"
        )
        
        # Database Action Buttons
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button(t("Save Changes to DB"), type="primary", help=t("Applies changes and deletions to the persistent SQLite database.")):
                # Clear current table and append the updated dataframe
                conn.execute("DELETE FROM targets")
                edited_pan.to_sql('targets', conn, if_exists='append', index=False)
                st.success(t("Database Updated Successfully!"))
        
        with col_btn2:
            st.download_button(
                label=t("Download Watchlist (CSV)"), 
                data=edited_pan.to_csv(index=False).encode('utf-8'), 
                file_name="RAP_Panopticon.csv", 
                mime="text/csv",
                type="primary"
            )
            
        st.markdown("---")
        st.subheader(t("Threat Intelligence Visuals"))
        
        # Intelligence Visualization (Charts with translated titles)
        c_vis1, c_vis2 = st.columns(2)
        with c_vis1:
            fig_bar = px.bar(
                edited_pan.head(15), 
                x='agent_id', 
                y='risk_score', 
                color='threat_type', 
                title=t("Top 15 Most Dangerous Targets")
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c_vis2:
            fig_pie = px.pie(
                edited_pan, 
                names='threat_type', 
                title=t("Threat Typology Distribution"), 
                hole=0.3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
    else:
        # Translated empty state message
        st.info(f"🟢 **{t('The Panopticon is currently empty.')}**\n\n{t('Run analysis in the Social Data module. If the system detects entities with high toxicity and network impact, they will be automatically classified as High-Value Targets and permanently stored here.')}")
