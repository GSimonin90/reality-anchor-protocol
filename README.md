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

### Features

1. **Multi-Vector Social Intelligence** Analyze public opinion from any source.
   * **Universal Import:** Ingest data from **Facebook, X (Twitter), Instagram, and TikTok** via CSV (compatible with extensions like *Instant Data Scraper*).
   * **Native YouTube Scraper:** Extract comments directly from video URLs without API keys.
   * **Mobile-Ready:** "Raw Text Paste" mode designed for mobile users to analyze copied threads on the fly.

2. **Hybrid Truth Engine (Logic + Facts + Emotion)** Powered by **Google Gemini 2.5**, the system performs a 3-layer deep scan:
   * **Logical Validity:** Detects formal and informal fallacies (e.g., *Ad Hominem, Strawman*).
   * **Factual Accuracy:** Flags historical inaccuracies, pseudoscience, and disinformation.
   * **Emotional Spectrum:** Measures **Sentiment** and **Aggression Levels** (0-10) to identify toxic patterns.

3. **Neural Logic Guard** Upload full **PDF documents** (contracts, articles, political programs) or long-form text. The AI scans the entire document to identify the *primary* logical flaws or factual errors undermining the central thesis.

4. **Reactive Simulation** Adjust parameters (*Bot Ratio, Influence Alpha, Agent Count*) on the fly and watch the "Belief Heatmap" evolve instantly in real-time to visualize the spread of misinformation.

5. **Enterprise Reporting** Generate professional, paginated **PDF Reports** containing executive summaries, fallacy distribution charts, word clouds, and detailed line-by-line analysis for research or journalism.

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


