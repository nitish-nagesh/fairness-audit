# 📄 app_lite.py
# Streamlit Lite Version - No R/Rpy2, Only Agent Explanation

import streamlit as st
import pyreadr
import tempfile
import openai
import os
from dotenv import load_dotenv

# Load environment variables (in local testing)
load_dotenv()

# OpenAI API key setup
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")



def explain_with_agent(text):
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a fairness-aware AI assistant who explains causal bias decomposition."},
            {"role": "user", "content": f"Here are results from a fairness audit:\n{text}\nExplain them in plain language."}
        ]
    )
    return response.choices[0].message.content

st.title("Causal Fairness Audit (Lite Version)")

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
        with st.spinner("Simulating fairness audit..."):
            # 📄 Simulated Fairness Decomposition Result (Mock Data)
            audit_result = '''
            tv: -0.0864 (±0.0161)
            ctfde: 0.0002 (±0.0128)
            ctfie: 0.0522 (±0.0066)
            ctfse: 0.0343 (±0.0111)
            ett: -0.0521 (±0.0119)
            '''
            st.session_state["audit_result"] = audit_result
            st.text("Fairness Decomposition Result:")
            st.code(audit_result, language="text")

    if st.button("Ask Agent to Explain"):
        if "audit_result" in st.session_state:
            with st.spinner("Calling GPT Agent..."):
                explanation = explain_with_agent(st.session_state["audit_result"])
                st.markdown("### Agent Explanation")
                st.markdown(explanation)
        else:
            st.warning("⚠️ Please run the fairness audit first.")