import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from duckduckgo_search import DDGS
from diseases import DISEASES

load_dotenv()

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

def generate_pdf_report(diagnosis_text, symptoms):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Clean markdown
    import re
    clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', diagnosis_text)
    clean_text = re.sub(r'\*(.*?)\*', r'\1', clean_text)
    
    # Title
    c.setFillColorRGB(0.1, 0.4, 0.7)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height-60, "AI Medical Diagnosis Report")
    
    # Line under title
    c.setLineWidth(2)
    c.line(50, height-70, width-50, height-70)
    
    # Date
    from datetime import datetime
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 10)
    c.drawString(50, height-90, f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # Symptoms
    c.setFillColorRGB(0.1, 0.4, 0.7)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height-120, "Symptoms Reported:")
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 11)
    c.drawString(50, height-140, symptoms[:80])
    
    # Diagnosis
    c.setFillColorRGB(0.1, 0.4, 0.7)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height-170, "Diagnosis Summary:")
    
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 10)
    y = height - 190
    
    for line in clean_text.split('\n'):
        if not line.strip():
            y -= 8
            continue
        # Word wrap
        words = line.split()
        current_line = ""
        for word in words:
            test = current_line + " " + word if current_line else word
            if c.stringWidth(test, "Helvetica", 10) < width - 100:
                current_line = test
            else:
                if y < 60:
                    c.showPage()
                    y = height - 50
                c.drawString(50, y, current_line)
                y -= 15
                current_line = word
        if current_line:
            if y < 60:
                c.showPage()
                y = height - 50
            c.drawString(50, y, current_line)
            y -= 15
    
    # Footer
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, 30, "This report is for informational purposes only. Always consult a qualified doctor.")
    
    c.save()
    buffer.seek(0)
    return buffer
# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY")
)

# Build Vector DB from diseases
@st.cache_resource
def build_vector_db():
    from rapidfuzz import fuzz, process
    disease_texts = []
    for disease in DISEASES:
        text = f"{disease['name']} {disease['symptoms']}"
        disease_texts.append((disease['name'], text))
    return disease_texts

# Agent State
class DiagnosisState(TypedDict):
    user_input: str
    extracted_symptoms: str
    retrieved_diseases: List[str]
    confidence_score: float
    retry_count: int
    medicines: str
    final_summary: str

# Agent 1: Symptom Extraction
def symptom_extraction_agent(state: DiagnosisState):
    prompt = f"""Extract specific medical symptoms from this user input.
User said: "{state['user_input']}"
Return only a comma-separated list of symptoms. Be specific and medical.
Example: fever, headache, nausea, fatigue"""
    
    response = llm.invoke(prompt)
    return {"extracted_symptoms": response.content}

# Agent 2: Retrieval Agent
def retrieval_agent(state: DiagnosisState):
    from rapidfuzz import process
    disease_texts = build_vector_db()
    query = state["extracted_symptoms"]
    results = process.extract(query, [text for _, text in disease_texts], limit=3)
    
    diseases = []
    confidence = 0
    for match, score, idx in results:
        diseases.append(disease_texts[idx][0])
        confidence = max(confidence, score / 100)
    
    return {
        "retrieved_diseases": diseases,
        "confidence_score": confidence
    }
# Agent 3: Corrective RAG - refines if confidence is low
def corrective_agent(state: DiagnosisState):
    if state["confidence_score"] < 0.3 and state["retry_count"] < 3:
        prompt = f"""The symptoms "{state['extracted_symptoms']}" gave low confidence results.
Rephrase and expand these symptoms with medical terminology to improve search.
Return only expanded symptom list."""
        response = llm.invoke(prompt)
        return {
            "extracted_symptoms": response.content,
            "retry_count": state["retry_count"] + 1
        }
    return state

# Check if retry needed
def should_retry(state: DiagnosisState):
    if state["confidence_score"] < 0.3 and state["retry_count"] < 3:
        return "retry"
    return "continue"

# Agent 4: Medicine Search Agent
def medicine_agent(state: DiagnosisState):
    try:
        disease_name = state["retrieved_diseases"][0].split("\n")[0].replace("Disease: ", "")
        with DDGS() as ddgs:
            results = list(ddgs.text(f"medicines treatment for {disease_name}", max_results=3))
        medicine_info =  "\n".join([r.get("body", "") for r in results[:2] if r.get("body")])
        return {"medicines": medicine_info}
    except:
        return {"medicines": "Please consult a doctor for medication advice."}

# Agent 5: Final Summarizer Agent
def summarizer_agent(state: DiagnosisState):
    prompt = f"""You are a medical AI assistant. Based on the vector database analysis, provide a structured diagnosis.

IMPORTANT: You MUST use the retrieved conditions from the vector database as your primary diagnosis. Do NOT suggest COVID-19 or other diseases unless they are in the retrieved conditions list below.

Patient symptoms: {state['user_input']}
Extracted symptoms: {state['extracted_symptoms']}
Retrieved conditions (USE THESE): {state['retrieved_diseases'][:2]}
Medicine info: {state['medicines']}

Provide:
1. Most likely condition (MUST be from Retrieved conditions above)
2. Why this matches symptoms
3. Severity level
4. Recommended action
5. Disclaimer
"""

    response = llm.invoke(prompt)
    return {"final_summary": response.content}

# Build LangGraph
@st.cache_resource
def build_graph():
    graph = StateGraph(DiagnosisState)
    
    graph.add_node("symptom_extraction", symptom_extraction_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("corrective", corrective_agent)
    graph.add_node("medicine_search", medicine_agent)
    graph.add_node("summarizer", summarizer_agent)
    
    graph.set_entry_point("symptom_extraction")
    graph.add_edge("symptom_extraction", "retrieval")
    graph.add_edge("retrieval", "corrective")
    graph.add_conditional_edges(
        "corrective",
        should_retry,
        {"retry": "retrieval", "continue": "medicine_search"}
    )
    graph.add_edge("medicine_search", "summarizer")
    graph.add_edge("summarizer", END)
    
    return graph.compile()

# Streamlit UI
st.set_page_config(page_title="AI Medical Diagnosis", page_icon="🏥", layout="wide")

# Sidebar
with st.sidebar:
    st.title("About")
    st.write("This AI assistant helps analyze symptoms and provides medical information.")
    st.warning("**Disclaimer:** This tool is for informational purposes only and should not replace professional medical advice.")
    st.markdown("---")
    # Symptom History
    st.markdown("### 🕐 Symptom History")
    if "symptom_history" not in st.session_state:
        st.session_state.symptom_history = []
    
    if st.session_state.symptom_history:
        for i, item in enumerate(reversed(st.session_state.symptom_history[-5:])):
            st.markdown(f"**{i+1}.** {item['symptom']}")
            st.caption(f"🔍 {item['diagnosis']} | {item['time']}")
            st.markdown("---")
    else:
        st.info("No history yet")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.symptom_history = []
        st.session_state.last_user_input = ""
        st.rerun()
st.title("🏥 Medical Diagnosis Assistant")
st.markdown("*AI-powered symptom analysis and medical information tool*")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Describe your symptoms (e.g., I have fatigue, cough, and fever)")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing symptoms..."):
            graph = build_graph()
            initial_state = {
                "user_input": user_input,
                "extracted_symptoms": "",
                "retrieved_diseases": [],
                "confidence_score": 0.0,
                "retry_count": 0,
                "medicines": "",
                "final_summary": ""
            }
            result = graph.invoke(initial_state)
            from datetime import datetime
            st.session_state.symptom_history.append({
                    "symptom": user_input,
                    "diagnosis": ', '.join(result.get('retrieved_diseases', ['Unknown'])),
                    "time": datetime.now().strftime("%H:%M")
    })
            response = result["final_summary"]
            severity_line = [line for line in response.split('\n') if 'severity' in line.lower()]
            severity_text = severity_line[0].lower() if severity_line else ""
            if any(word in severity_text for word in ["high", "severe", "critical", "emergency", "moderate to severe"]):
                st.error("🔴 SEVERITY: HIGH — Please seek immediate medical attention!")
            elif any(word in severity_text for word in ["moderate"]):
                st.warning("🟡 SEVERITY: MODERATE — Consult a doctor soon")
            else:
                st.success("🟢 SEVERITY: MILD — Monitor symptoms at home")
            st.markdown(response)

            with st.expander("Processing Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Extracted Symptoms")
                    st.write(result.get("extracted_symptoms", ""))
                with col2:
                    st.subheader("Analysis Details")
                    st.write(f"**Similarity Score:** {result.get('confidence_score', 0):.2f}")
                    st.write(f"**Retry Count:** {result.get('retry_count', 0)}")
                    st.write(f"**Retrieved Disease:** {', '.join(result.get('retrieved_diseases', []))}")

            st.session_state['last_user_input'] = user_input
            # PDF Download Button
        pdf_buffer = generate_pdf_report(
            response, 
            user_input
        )
        st.download_button(
            label="📄 Download Diagnosis Report (PDF)",
            data=pdf_buffer,
            file_name="diagnosis_report.pdf",
            mime="application/pdf"
        )

    if 'response' in locals():
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if 'last_user_input' in st.session_state:
    if st.button("💊 Get Medicine Information", key="med_btn"):
        with st.spinner("Searching medicines..."):
            try:
                with DDGS() as ddgs:
                    query = st.session_state['last_user_input']
                    med_results = list(ddgs.text(f"medicines treatment for {query}", max_results=3))
                if med_results:
                    st.success("💊 Medicine Information:")
                    for r in med_results[:2]:
                        st.write(r.get("body", ""))
                else:
                    st.info("Please consult a doctor for medication advice.")
            except:
                st.info("Please consult a doctor for medication advice.")
    if 'response' in locals():
        st.session_state.chat_history.append({"role": "assistant", "content": response})