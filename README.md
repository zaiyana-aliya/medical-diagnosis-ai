# 🏥 AI Medical Diagnosis Assistant

> An intelligent medical symptom analysis chatbot powered by **Agentic RAG**, **LangGraph**, **Groq LLaMA 3**, and **Streamlit**.

[Python](https://img.shields.io/badge/Python-3.12-blue)
[Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
[LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-green)
[Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange)

---

## 🚀 Features

- 🧠 **AI-Powered Symptom Analysis** — Uses Groq LLaMA 3.3 70B for intelligent diagnosis
- 🔄 **Agentic RAG Pipeline** — 5-agent LangGraph workflow for accurate results
- 🔁 **Corrective RAG** — Auto-refines query up to 3 times if confidence is low
- 💊 **Medicine Information** — Real-time medicine search via DuckDuckGo API
- 🚨 **Severity Detection** — Classifies as MILD / MODERATE / HIGH
- 📄 **PDF Report Download** — Professional diagnosis report generation
- 🕐 **Symptom History** — Tracks all past diagnoses in sidebar
- 🗄️ **200+ Diseases Database** — Covers respiratory, cardiac, neurological, tropical & more

---

## 🏗️ Architecture

```
User Input
    ↓
Agent 1: Symptom Extraction (Groq LLaMA)
    ↓
Agent 2: Disease Retrieval (Vector DB - 200+ diseases)
    ↓
Agent 3: Corrective RAG (refines if confidence < 0.3)
    ↓
Agent 4: Medicine Search (DuckDuckGo API)
    ↓
Agent 5: Final Summarizer (Structured Diagnosis Report)
    ↓
Streamlit UI (Severity Badge + PDF Download)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq LLaMA 3.3 70B Versatile |
| Agent Framework | LangGraph |
| Vector Search | ChromaDB + RapidFuzz |
| Medicine Search | DuckDuckGo API |
| Frontend | Streamlit |
| PDF Generation | ReportLab |
| Embeddings | FakeEmbeddings (prototype) |

---

## 📋 Disease Categories

- 🫁 Respiratory (Asthma, Pneumonia, TB, COPD)
- ❤️ Cardiovascular (Heart Attack, Stroke, DVT)
- 🧠 Neurological (Migraine, Epilepsy, Meningitis)
- 🦠 Infectious (Dengue, Malaria, Chikungunya, Typhoid)
- 🏥 Gastrointestinal (IBS, Gastritis, Appendicitis)
- 🩺 Endocrine (Diabetes, Hypothyroidism)
- 🧬 Skin (Eczema, Psoriasis, Cellulitis)
- 🧠 Mental Health (Depression, Anxiety, Bipolar)
- 🌴 Kerala-specific (Nipah, Leptospirosis, Scrub Typhus)
- And 150+ more!

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/zaiyana-aliya/medical-diagnosis-ai.git
cd medical-diagnosis-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create `.env` file
```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
SERPER_API_KEY=your_serper_api_key
SERPAPI_API_KEY=your_serpapi_api_key
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🔑 API Keys Required

| API | Free Tier | Link |
|-----|-----------|------|
| Groq | ✅ Free | [console.groq.com](https://console.groq.com) |
| Gemini | ✅ Free | [aistudio.google.com](https://aistudio.google.com) |
| Serper | ✅ 2500 free | [serper.dev](https://serper.dev) |
| SerpAPI | ✅ 250/month | [serpapi.com](https://serpapi.com) |

---

## 📸 Screenshots

### Severity Detection
- 🟢 **MILD** — Monitor symptoms at home
- 🟡 **MODERATE** — Consult a doctor soon  
- 🔴 **HIGH** — Seek immediate medical attention

### PDF Report
Professional diagnosis report with symptoms, condition, severity and recommendations.

---

## ⚠️ Disclaimer

This tool is for **informational purposes only** and should **not replace professional medical advice**. Always consult a qualified healthcare professional for accurate diagnosis and treatment.

---

## 👩‍💻 Developer

**Zaiyana Aliya**  
Project Coordinator & Business Analyst | Healthcare IT  
📍 Palakkad, Kerala, India  
🔗 [LinkedIn](https://linkedin.com/in/zaiyana-aliya-612004273) | [GitHub](https://github.com/zaiyana-aliya)

---

## 📄 License

MIT License — feel free to use and modify for educational purposes.
