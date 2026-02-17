# The Reality Anchor Protocol (R.A.P.) simulations
**An Agent-Based Model investigation into Disinformation and Cognitive Resilience**

> *"Does perfect algorithmic safety inevitably lead to human cognitive atrophy?"*

## Project Overview
This repository hosts the Python simulation code and research papers produced by the **Computational Social Simulation Lab**. 

We set out to model a digital society ($N=1000$ agents) resistant to AI-generated disinformation. While testing a new technical architecture, our simulations uncovered a counter-intuitive psychological paradox regarding human reliance on automation.

## Documentation
* **[Part 1: The Architecture (PDF)](docs/Reality_Anchor_Protocol.pdf)** - A proposal for BFT-compliant social ranking using C2PA and Bridging Algorithms.
* **[Part 2: The Cognitive Risk (PDF)](docs/R.A.P._Addendum.pdf)** - Analysis of the "Epistemic Atrophy" phenomenon observed in simulations.

---

## Key Findings

### 1. The Technical Experiment (Resilience)
We modeled a network under attack by bot farms (>40% malicious nodes). We tested a "Defense in Depth" architecture combining:
* **Hardware Provenance:** Simulating C2PA/Secure Enclave checks.
* **Bridging-Based Ranking:** An algorithm that prioritizes cross-cluster consensus over raw engagement.

![R.A.P. Resilience Heatmap](output/fig1_resilience.png)
*Fig 1: Heatmap showing agent beliefs over time. Despite 40% malicious bots (Red block), the bridging algorithm prevents total contagion in the general population (Striped section).*

**Result:** The simulation suggests that replacing virality with bridging scores allows the network to maintain bounded consensus ($\Delta \le 0.47$) even under high stress conditions.

### 2. The Psychological Paradox (Atrophy)
During the stress tests, we tracked a variable representing human **Critical Skill** ($S$).

![Cognitive Atrophy vs Socratic Engine](output/fig2_socratic_atrophy.png)
*Fig 2: Comparison of human critical skills over time. The Red line shows rapid decay under full automation. The Blue line shows sustained skill levels using the Socratic Engine.*

* **Observation:** When the algorithmic filters worked perfectly (100% protection), the simulated human agents reduced their verification efforts. Over time, $S$ decayed to near zero ($<0.10$), a state we call "Epistemic Atrophy".
* **Hypothesis:** A system that removes all friction may make the network technically safe but the users cognitively vulnerable.

### 3. Proposed Countermeasure: "Socratic Friction"
We tested a modified engine (Layer 2.5) that intentionally re-introduces friction.
* By challenging the user with verification tasks at a calculated rate (~40%), the simulation showed that agents maintained high critical skills ($S > 0.85$) without compromising system stability.

---

## Interactive Dashboard (https://reality-anchor-protocol-dashboard.streamlit.app/)

We have developed a real-time web interface to interact with the models and visualize real-world data.

# Features

### Social Data Analysis (The Core)
* **Universal Import:** Ingest data from **Facebook, X (Twitter), Instagram, and TikTok** via CSV (compatible with extensions like *Instant Data Scraper*).
* **Native YouTube Scraper:** Extract comments directly from video URLs without requiring personal API keys.
* **Adaptive Personas:** Switch the analytical lens instantly. Ask the AI to analyze data as a **Strategic Intelligence Analyst**, a **Mass Psychologist** (focus on emotional triggers), a **Legal Consultant** (focus on liability/defamation), or a **Campaign Manager** (focus on opportunity).
* **The Oracle:** Chat directly with your dataset. Ask questions like *"Who is the main target of hate?"* or *"Find the most persuasive argument"* and get answers based on the analyzed data.

### Comparison Test (A/B Testing)
* **Head-to-Head Comparison:** Compare two different datasets or YouTube videos side-by-side.
* **Metric Battle:** Instantly visualize the difference in **Aggression Levels**, **Bot Presence**, and **Logical Fallacies** between two sources (e.g., "Right-wing Video" vs "Left-wing Video").

### Wargame Room (Live Simulation)
* **Real-Time Info-War Simulator:** Visualize the spread of disinformation with a reactive heatmap and infection graphs.
* **Network Topologies:** Choose the battlefield scenario: **Public Square** (high connectivity), **Echo Chambers** (isolated clusters), or **Influencer Network** (central hubs).
* **Blue Team Countermeasures:** Deploy active defenses like **Targeted Debunking**, **Algorithmic Dampening**, or **Hard Bans** and watch the infection rate curve change in real-time.

### Cognitive Editor & Vision Guard (Multimodal)
* **Vision Guard:** Upload **Images, Memes, or Screenshots**. The AI "sees" the image, analyzes visual symbolism/text, and detects manipulative patterns or fallacies.
* **Voice Intel:** Upload **Audio files (MP3, WAV, M4A)**. The AI "listens" to the speech, analyzing **Tone, Prosody, and Emotion** to detect sarcasm, hesitation, or aggression that text analysis misses.
* **Auto-Sanitize:** The system doesn't just flag errors; it **rewrites** toxic or fallacious text into a neutral, logical, and factual version.
* **Fact Extraction:** Automatically extracts a bulleted list of factual claims from Text, PDFs, or Images that require external verification.
* **Document Analysis:** Upload full **PDF documents** (contracts, articles, political programs) for deep logical scanning.

### Enterprise Reporting
* **Hybrid Truth Engine:** Powered by **Google Gemini 2.0 Flash**, performing a 3-layer deep scan (Logical Validity, Factual Accuracy, Emotional Spectrum).
* **Bot Hunter:** Heuristic detection of coordinated inauthentic behavior and spam bots.
* **Export Ready:** Generate professional **Excel** datasets or paginated **PDF Reports** containing executive summaries, fallacy distribution charts, and word clouds for research or journalism.

### How to Run Locally

    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Launch the Dashboard
    streamlit run app.py

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `app.py` | **Main Dashboard application**. Interactive Streamlit interface. |
| `moltbook_REAL_data.csv` | Dataset containing real agent interactions from the Moltbook archive. |
| `src/01_rap_bft_simulation.py` | Simulates the network resilience against bot swarms (Hegselmann-Krause dynamics). |
| `src/02_cognitive_atrophy.py` | Models the decay of human critical skills under total automation vs. Socratic Friction. |
| `download_moltbook.py` | Script to fetch and clean fresh data from the Moltbook Observatory. |
| `output/` | Contains the generated graphs (`fig1`, `fig2`). |

---

## Contributing

This is an experimental research project. We welcome forks, pull requests, and alternative modeling approaches to challenge our parameters.




