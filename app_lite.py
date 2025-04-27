import streamlit as st
import pyreadr
import tempfile
import openai
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import re
import base64

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

# --- UI ---
st.title("Causal Fairness Audit (Lite Version)")

# (Optional) Upload .rda file (even if not used in this version)
uploaded_file = st.file_uploader("Upload COMPAS .rda file (Optional)", type="rda")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".rda") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    st.write("### Sample of Uploaded Data")
    result = pyreadr.read_r(tmp_path)
    df = result[list(result.keys())[0]]
    st.dataframe(df.head())

# 📄 Hardcoded simulated audit results
audit_result = '''
tv: -0.0864 (±0.0161)
ctfde: 0.0002 (±0.0128)
ctfie: 0.0522 (±0.0066)
ctfse: 0.0343 (±0.0111)
ett: -0.0521 (±0.0119)
'''

if st.button("Run Fairness Audit (Simulated)"):
    with st.spinner("Parsing fairness audit results..."):
        st.text("Fairness Decomposition Result:")
        st.code(audit_result, language="text")

        # 📊 Parse and plot the audit result
        pattern = r"(\w+): ([\-\d.]+) \(±([\d.]+)\)"
        matches = re.findall(pattern, audit_result)
        if matches:
            plot_df = pd.DataFrame(matches, columns=["measure", "value", "sd"])
            plot_df["value"] = plot_df["value"].astype(float)
            plot_df["sd"] = plot_df["sd"].astype(float)

            fig, ax = plt.subplots()
            colors = ["salmon", "olive", "skyblue", "orchid", "lightgrey"]
            ax.bar(plot_df["measure"], plot_df["value"], yerr=plot_df["sd"], capsize=8, color=colors[:len(plot_df)])
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_ylabel("Value")
            ax.set_title("Simulated Y disparity decomposition (Sample Audit)")
            st.pyplot(fig)

    # Save audit result for agent
    st.session_state["audit_result"] = audit_result

if st.button("Ask Agent to Explain"):
    if "audit_result" in st.session_state:
        with st.spinner("Calling GPT Agent..."):
            explanation = explain_with_agent(st.session_state["audit_result"])
            st.markdown("### Agent Explanation")
            st.markdown(explanation)
    else:
        st.warning("⚠️ Please run the fairness audit first.")

if st.button("Run Prediction and Show Fairness Plot"):
    st.markdown("### Fairness Decomposition Plot (Random Forest Predictions)")
    st.image("fig_compas_yhat_rf.png", use_container_width=True, caption="COMPAS Fairness Decomposition after Random Forest prediction")
    
if st.button("Ask GPT-4o to Explain Prediction Plot"):
    with st.spinner("Calling GPT-4o Vision..."):
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        # Read image file and encode it as base64
        with open("fig_compas_yhat_rf.png", "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()

        # Call GPT-4o with image
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a fairness-aware AI assistant who explains fairness decomposition plots for machine learning models."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please explain this fairness decomposition plot clearly. Focus on treatment effects and bias components."},
                        {"type": "image", "image": {"base64": encoded_image}}
                    ]
                }
            ]
        )

        plot_explanation = response.choices[0].message.content

        st.markdown("### GPT-4o Explanation for Prediction Plot")
        st.markdown(plot_explanation)