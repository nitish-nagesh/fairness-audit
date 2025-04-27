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
    
import base64

random_forest_prompt = """
You are analyzing a fairness decomposition plot from a machine learning study.

This plot shows two bars for each fairness component:
- **Original** outcome disparities (red bars)
- **Predicted** outcome disparities after applying a Random Forest model (light red/pink bars)

The x-axis shows the following components:
- Total Variation (tv)
- Conditional Treatment-Free Direct Effect (ctfde)
- Conditional Treatment-Free Indirect Effect (ctfie)
- Conditional Total Sequential Effect (ctfse)
- Effect of Treatment on Base Inputs (ett)

The y-axis shows the magnitude of bias, positive or negative, with error bars.

Please explain:
- What each component (tv, ctfde, ctfie, etc.) represents
- How the bias has changed between Original and Predicted (has Random Forest reduced or worsened bias?)
- Which biases are most affected by the Random Forest predictions
- Overall, does the Random Forest model seem to mitigate or amplify fairness issues compared to the original data?

Focus on comparing the **Original vs Predicted** results and give a fairness-aware interpretation.
"""

if st.button("Ask GPT-4o to Explain Prediction Plot"):
    with st.spinner("Calling GPT-4o Vision..."):
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        # Read image file and encode as base64
        with open("fig_compas_yhat_rf.png", "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode()

        # Create the correct format: data URL
        data_url = f"data:image/png;base64,{image_base64}"

        # Call GPT-4o with proper image_url format
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a fairness-aware AI assistant who explains fairness decomposition plots for machine learning models."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": random_forest_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]
        )

        plot_explanation = response.choices[0].message.content

        st.markdown("### GPT-4o Explanation for Prediction Plot")
        st.markdown(plot_explanation)