# 🤖 Smart SRE: AI-Driven Incident Response Agent

### Overview
**Smart SRE** is a specialized RAG (Retrieval-Augmented Generation) prototype designed to reduce **Mean Time To Resolution (MTTR)** for DevOps teams. 

Instead of manually searching through thousands of logs, this agent uses **Semantic Search (TF-IDF & Cosine Similarity)** to instantly map current production errors to historical incidents, providing root causes and proven fixes in milliseconds.

### Architecture
* **Core Logic:** Vector Space Modeling (TF-IDF) using Scikit-Learn.
* **Search Algorithm:** Cosine Similarity for semantic matching.
* **Intelligence Layer:** Auto-categorization of severity, team assignment, and cost estimation.
* **Privacy:** Runs 100% offline (No data leaves the perimeter).

### Key Features
* **Context-Aware Retrieval:** Understands that "Slow API" is related to "Latency" and "Timeout".
* **Confidence Scoring:** Assigns a probability score to matches to prevent false positives.
* **Business Impact Analysis:** Automatically estimates potential financial risk ($/hr) based on severity.
* **Auto-Routing:** Classifies incidents for Database, Network, or SRE teams.

### How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the agent: `python rca_agent.py`

---
*Developed as a Strategic Initiative to modernize Incident Management workflows.*
