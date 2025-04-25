### 1. app.py (Main Streamlit App)
import streamlit as st
import tempfile
import subprocess
import openai
import pyreadr
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI()

def explain_with_agent(text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a fairness-aware AI assistant who explains causal bias decomposition."},
            {"role": "user", "content": f"Here are results from a fairness audit:\n{text}\nExplain them in plain language."}
        ]
    )
    return response.choices[0].message.content

st.title("Causal Fairness Audit (Agent-Enhanced)")

uploaded_file = st.file_uploader("Upload COMPAS .rda file", type="rda")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".rda") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    st.write("### Sample of Uploaded Data")
    result = pyreadr.read_r(tmp_path)
    df = result[list(result.keys())[0]]
    st.dataframe(df.head())

    if st.button("Run Fairness Audit"):
        with st.spinner("Running fairness audit via external script..."):
            try:
                subprocess.run(["python3", "fairness_audit_runner.py", tmp_path], check=True)
                with open("audit_output.txt", "r") as f:
                    audit_result = f.read()
                st.session_state["audit_result"] = audit_result
                st.text("Fairness Decomposition Result:")
                st.code(audit_result, language="text")
            except Exception as e:
                st.error(f"🚨 Failed to run fairness audit: {e}")

    if st.button("Ask Agent to Explain"):
        if "audit_result" in st.session_state:
            with st.spinner("Calling GPT Agent..."):
                explanation = explain_with_agent(st.session_state["audit_result"])
                st.markdown("### Agent Explanation")
                st.markdown(explanation)
        else:
            st.warning("⚠️ Please run the fairness audit first.")
