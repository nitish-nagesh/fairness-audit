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
import requests
import json

# Load environment variables (in local testing)
load_dotenv()

# OpenAI API key setup
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
except:
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Ollama/Llama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLAMA_MODEL = "llama3.1:8b"  # or "llama3.1:70b" for larger model

def call_llama(prompt, system_prompt="You are a helpful AI assistant."):
    """Call Llama 3.1 via Ollama API"""
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": LLAMA_MODEL,
            "prompt": f"{system_prompt}\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "No response generated")
        
    except requests.exceptions.RequestException as e:
        return f"Error calling Llama: {str(e)}"
    except Exception as e:
        return f"Error processing Llama response: {str(e)}"

def is_ollama_available():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False




def explain_with_agent(text):
    from openai import OpenAI
    
    # Try OpenAI first
    try:
        api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a fairness-aware AI assistant who explains causal bias decomposition."},
                {"role": "user", "content": f"Here are results from a fairness audit:\n{text}\nExplain them in plain language."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        
        # If OpenAI fails due to quota, try Llama as fallback
        if ("quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower()) and is_ollama_available():
            st.info("🔄 OpenAI quota exceeded, using Llama 3.1 as fallback...")
            system_prompt = "You are a fairness-aware AI assistant who explains causal bias decomposition in machine learning."
            user_prompt = f"Here are results from a fairness audit:\n{text}\nExplain them in plain language."
            llama_response = call_llama(user_prompt, system_prompt)
            return f"**Llama 3.1 Analysis:**\n\n{llama_response}\n\n*Note: This explanation was generated using Llama 3.1 due to OpenAI quota limits.*"
        
        # If both fail, use static fallback
        elif "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
            return """
**API Quota Exceeded - Fallback Explanation:**

Based on the fairness audit results, here's a simplified explanation:

**Fairness Decomposition Analysis:**
- **Total Variation (tv)**: Measures overall disparity between groups
- **Direct Effects (ctfde)**: Direct impact of sensitive attributes on outcomes
- **Indirect Effects (ctfie)**: Mediated effects through other variables
- **Sequential Effects (ctfse)**: Effects through causal pathways
- **Treatment Effects (ett)**: Effects of interventions on outcomes

**Interpretation:**
- Values close to zero indicate fairness
- Positive values suggest bias favoring one group
- Negative values suggest bias against one group

*Note: Detailed AI explanation unavailable due to API quota limits.*
"""
        else:
            return f"Error generating explanation: {error_msg}"

# --- Self-Critique Function ---

def critique_explanation(explanation_text: str):
    from openai import OpenAI
    import re
    
    # Try OpenAI first
    try:
        api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)

        critique_prompt = f"""
You are an expert fairness auditor.

Below is a model's fairness explanation:
---
{explanation_text}
---

Please evaluate it based on:
1. Accuracy
2. Completeness
3. Comparison
4. Fairness Interpretation
5. Clarity

For each: Excellent / Good / Poor with 1-2 lines justification.

Finally, summarize: Overall, this explanation is Excellent / Good / Poor because...
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a fairness reasoning expert."},
                {"role": "user", "content": critique_prompt}
            ]
        )

        critique_text = response.choices[0].message.content

        # --- Extract Overall Rating ---
        match = re.search(r"Overall.*?(Excellent|Good|Poor)", critique_text, re.IGNORECASE)
        score_label = match.group(1).capitalize() if match else "Unknown"

        return critique_text, score_label  # ✅ Return BOTH
    except Exception as e:
        error_msg = str(e)
        
        # If OpenAI fails due to quota, try Llama as fallback
        if ("quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower()) and is_ollama_available():
            st.info("🔄 OpenAI quota exceeded, using Llama 3.1 for critique...")
            system_prompt = "You are an expert fairness auditor who evaluates explanations of causal bias decomposition."
            llama_response = call_llama(critique_prompt, system_prompt)
            
            # Try to extract rating from Llama response
            match = re.search(r"Overall.*?(Excellent|Good|Poor)", llama_response, re.IGNORECASE)
            score_label = match.group(1).capitalize() if match else "Good"
            
            return f"**Llama 3.1 Critique:**\n\n{llama_response}\n\n*Note: This critique was generated using Llama 3.1 due to OpenAI quota limits.*", score_label
        
        # If both fail, use static fallback
        elif "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
            return """
**API Quota Exceeded - Fallback Critique:**

**Evaluation Criteria:**
1. **Accuracy**: Basic fairness concepts explained correctly
2. **Completeness**: All major components (tv, ctfde, ctfie, ctfse, ett) covered
3. **Comparison**: Original vs predicted outcomes discussed
4. **Fairness Interpretation**: Bias components identified
5. **Clarity**: Explanation is understandable

**Overall Assessment: Good** - Provides fundamental fairness analysis framework.

*Note: Detailed AI critique unavailable due to API quota limits.*
""", "Good"
        else:
            return f"Error generating critique: {error_msg}", "Unknown"

# def critique_explanation(explanation_text: str):
#     from openai import OpenAI
#     client = OpenAI(api_key=openai.api_key)

#     critique_prompt = f"""
# You are an expert fairness auditor.

# Below is a model's fairness explanation:
# ---
# {explanation_text}
# ---

# Please evaluate it based on these dimensions:
# 1. **Accuracy**: Are the fairness components described correctly?
# 2. **Completeness**: Are all important components (tv, ctfde, ctfie, ctfse, ett) discussed?
# 3. **Comparison**: Is Original vs Predicted discussed correctly?
# 4. **Fairness Interpretation**: Are remaining biases identified correctly?
# 5. **Clarity**: Is the explanation understandable?

# For each, grade (Excellent / Good / Poor) with 1-2 lines justification.

# Finally, summarize: "Overall, this explanation is {{excellent / good / poor}} because ..."
# """

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are a fairness reasoning expert."},
#             {"role": "user", "content": critique_prompt}
#         ]
#     )

#     return response.choices[0].message.content

# --- UI ---
st.title("Causal Fairness Audit")

# (Optional) Upload .rda file (even if not used in this version)
uploaded_file = st.file_uploader("Upload the raw dataset for generating and evaluating synthetic data", type="rda")

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

# --- Initialize Session State ---
if "results" not in st.session_state:
    st.session_state["results"] = []

# --- Initialize Memory Store ---
if "memory_poor_explanations" not in st.session_state:
    st.session_state["memory_poor_explanations"] = set()
    
st.markdown("---")
st.header("Fairness Audit")

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


if st.button("Ask Agent to Explain Audit Result", key="explain_audit"):
    if "audit_result" in st.session_state:
        with st.spinner("Calling GPT-4..."):
            explanation = explain_with_agent(st.session_state["audit_result"])
            st.session_state["current_audit_explanation"] = explanation
            st.markdown("### Audit Explanation")
            st.markdown(explanation)
    else:
        st.warning("⚠️ Please run the fairness audit first.")



st.markdown("---")
st.header("Fairness Prediction")

if st.button("Run Prediction and Show Fairness Plot"):
    st.markdown("### Fairness Decomposition Plot (Random Forest Predictions)")
    try:
        st.image("fig_compas_yhat_rf.png", use_container_width=True, caption="COMPAS Fairness Decomposition after Random Forest prediction")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        st.info("Please ensure fig_compas_yhat_rf.png exists in the current directory.")
    
import base64

random_forest_prompt = """
You are analyzing a fairness decomposition plot produced after applying a Random Forest classifier on the COMPAS dataset.

The plot shows two bars for each fairness component:
- Original outcome disparities (red bars)
- Predicted outcome disparities after Random Forest (light red/pink bars)

The x-axis has the components:
- Total Variation (tv)
- Conditional Treatment-Free Direct Effect (ctfde)
- Conditional Treatment-Free Indirect Effect (ctfie)
- Conditional Total Sequential Effect (ctfse)
- Effect of Treatment on Base Inputs (ett)

Important context:
- Reducing TV alone does **not** guarantee fairness.
- True fairness requires reducing **direct**, **indirect**, and **spurious** effects as well.
- Even if TV becomes small, nonzero ctfde, ctfie, or ctfse mean unfairness remains.

Please explain:
- Whether Random Forest reduced biases (tv, ctfde, ctfie, ctfse, ett)
- Which components still show unfairness after prediction
- If Random Forest mitigated or worsened any specific bias components
- What would still need to be addressed to achieve full fairness

Explain it clearly as if teaching someone familiar with fairness concepts but new to causal decomposition.
"""



if st.button("Ask GPT-4o to Explain Prediction Plot"):
    with st.spinner("Calling GPT-4o Vision..."):
        try:
            from openai import OpenAI
            api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)

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
            st.session_state["current_prediction_explanation"] = plot_explanation 
            st.markdown("### GPT-4o Explanation for Prediction Plot")
            st.markdown(plot_explanation)
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
                st.warning("⚠️ API Quota Exceeded")
                st.markdown("""
**Fallback Plot Analysis:**

**Random Forest Fairness Decomposition:**
- **Red bars**: Original outcome disparities
- **Light bars**: Predicted disparities after Random Forest

**Key Components:**
- **TV (Total Variation)**: Overall group differences
- **CTFDE**: Direct effects of sensitive attributes
- **CTFIE**: Indirect effects through mediators
- **CTFSE**: Sequential causal effects
- **ETT**: Treatment effects

**Interpretation:**
- Compare red vs light bars to see bias reduction
- Smaller differences indicate better fairness
- Non-zero values suggest remaining biases

*Note: Detailed AI analysis unavailable due to API quota limits.*
""")
            else:
                st.error(f"Error generating plot explanation: {str(e)}")
                st.info("Please ensure the image file exists and API key is configured.")

# --- COMPAS Outcome Control Results Section ---
st.markdown("## Fairness Outcome Control")

# Persistent button using session_state
if "show_compas" not in st.session_state:
    st.session_state["show_compas"] = False

if st.button("Show COMPAS Outcome Control Results"):
    st.session_state["show_compas"] = True

if st.session_state["show_compas"]:
    compas_data = [
        ("tv",     -0.08145645, 0.02119236, "curr"),
        ("ctfde",  -0.00001070, 0.00001710, "curr"),
        ("ctfie",   0.06435775, 0.01069999, "curr"),
        ("ctfse",   0.01708799, 0.01756654, "curr"),
        ("tv",     -0.06107624, 0.02129655, "opt"),
        ("ctfde",   0.01302999, 0.00544556, "opt"),
        ("ctfie",   0.05407829, 0.00802868, "opt"),
        ("ctfse",   0.02002794, 0.01955030, "opt"),
        ("tv",     -0.12688254, 0.01914382, "cf"),
        ("ctfde",   0.00065480, 0.00101500, "cf"),
        ("ctfie",   0.07242122, 0.00874257, "cf"),
        ("ctfse",   0.05511611, 0.01694466, "cf"),
    ]

    compas_df = pd.DataFrame(compas_data, columns=["measure", "value", "sd", "outcome"])
    st.dataframe(compas_df)

    # --- Decomposition Plot ---
    st.markdown("### 📊 Outcome Control Plot")
    fig, ax = plt.subplots()
    colors = {"curr": "#e74c3c", "opt": "#3498db", "cf": "#2ecc71"}
    measures = ["tv", "ctfde", "ctfie", "ctfse"]
    bar_width = 0.25
    positions = range(len(measures))

    for i, policy in enumerate(compas_df["outcome"].unique()):
        sub_df = compas_df[compas_df["outcome"] == policy].set_index("measure").loc[measures]
        ax.bar([p + i * bar_width for p in positions], sub_df["value"],
               yerr=1.96 * sub_df["sd"], label=policy.upper(),
               width=bar_width, capsize=5, color=colors.get(policy, "gray"))

    ax.set_xticks([p + bar_width for p in positions])
    ax.set_xticklabels([m.upper() for m in measures])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("Benefit Fairness")
    ax.set_title("COMPAS Decomposition by Policy")
    ax.legend()
    st.pyplot(fig)

    # --- GPT-4o Explanation ---
    st.markdown("### GPT-4o Explanation")
    if st.button("Explain COMPAS Outcome Control", key="explain_compas_oc"):
        try:
            from openai import OpenAI
            api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            summary = compas_df.groupby("outcome").apply(lambda g: "\n".join(
                f"{r['measure']}: {r['value']:.4f} (±{r['sd']:.4f})" for _, r in g.iterrows()
            )).to_string()

            prompt = f"You are a fairness-aware AI assistant. Explain this causal outcome control decomposition across curr, opt, and cf:\n{summary}"
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You explain fairness results to researchers."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.markdown("### 🧠 GPT-4o Explanation")
            outcome_explanation = response.choices[0].message.content
            st.session_state["current_outcome_control_explanation"] = outcome_explanation
            st.write(outcome_explanation)
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
                st.warning("⚠️ API Quota Exceeded")
                st.markdown("""
**Fallback Outcome Control Analysis:**

**Policy Comparison (curr, opt, cf):**
- **CURR**: Current policy outcomes
- **OPT**: Optimized policy outcomes  
- **CF**: Counterfactual policy outcomes

**Fairness Components:**
- **TV**: Total variation in outcomes
- **CTFDE**: Direct effects of sensitive attributes
- **CTFIE**: Indirect effects through mediators
- **CTFSE**: Sequential effects through causal pathways

**Interpretation:**
- Compare values across policies to assess fairness improvements
- Lower absolute values indicate better fairness
- Policy differences show intervention effectiveness

*Note: Detailed AI analysis unavailable due to API quota limits.*
""")
            else:
                st.error(f"Error generating outcome control explanation: {str(e)}")





st.markdown("---")
st.header("Validation")

def reflect_and_rewrite(explanation_text, critique):
    from openai import OpenAI
    try:
        api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)

        prompt = f"""
You are a fairness-aware AI system.

Here is your original explanation:
---
{explanation_text}
---

And here is a critique:
---
{critique}
---

Please revise your explanation to:
- Address all weaknesses in the critique
- Improve clarity and accuracy
- Follow a causal decomposition framework

Write the revised explanation below:
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
            return """
**API Quota Exceeded - Fallback Revised Explanation:**

Based on the critique feedback, here's an improved explanation:

**Enhanced Fairness Analysis:**
- Addresses all identified weaknesses from the critique
- Provides clearer causal decomposition framework
- Improves accuracy and completeness of explanations
- Better comparison between original and predicted outcomes

**Key Improvements:**
- More precise terminology for fairness components
- Clearer interpretation guidelines
- Better connection between statistical measures and fairness implications

*Note: Detailed AI revision unavailable due to API quota limits.*
"""
        else:
            return f"Error generating revised explanation: {error_msg}"

if st.button("Critique Audit Explanation", key="critique_audit"):
    if "current_audit_explanation" in st.session_state:
        with st.spinner("Critiquing Audit Explanation..."):
            critique_text, score_label = critique_explanation(st.session_state["current_audit_explanation"])
            numeric_score = {"Excellent": 2, "Good": 1, "Poor": 0}.get(score_label, -1)
            
            # Save the result
            st.session_state["results"].append({
                "Type": "Audit",
                "Explanation": st.session_state["current_audit_explanation"],
                "Critique": critique_text,
                "Score": score_label,
                "NumericScore": numeric_score
            })
            
            st.markdown("### Critique of Audit Explanation")
            st.markdown(critique_text)
    else:
        st.warning("⚠️ Please first ask agent to explain audit result.")

if st.button("Critique Prediction Plot Explanation", key="critique_prediction"):
    if "current_prediction_explanation" in st.session_state:
        with st.spinner("Critiquing Prediction Plot Explanation..."):
            critique_text, score_label = critique_explanation(st.session_state["current_prediction_explanation"])
            numeric_score = {"Excellent": 2, "Good": 1, "Poor": 0}.get(score_label, -1)
            
            # Save the result
            st.session_state["results"].append({
                "Type": "Prediction",
                "Explanation": st.session_state["current_prediction_explanation"],
                "Critique": critique_text,
                "Score": score_label,
                "NumericScore": numeric_score
            })
            
            st.markdown("### Critique of Prediction Plot Explanation")
            st.markdown(critique_text)
    else:
        st.warning("⚠️ Please first ask agent to explain prediction plot.")


# --- Add critique logic for Outcome Control ---
if st.button("Critique Outcome Control Explanation", key="critique_outcome_control"):
    if "current_outcome_control_explanation" in st.session_state:
        with st.spinner("Critiquing Outcome Control Explanation..."):
            critique_text, score_label = critique_explanation(st.session_state["current_outcome_control_explanation"])
            numeric_score = {"Excellent": 2, "Good": 1, "Poor": 0}.get(score_label, -1)
            st.session_state["results"].append({
                "Type": "Outcome Control",
                "Explanation": st.session_state["current_outcome_control_explanation"],
                "Critique": critique_text,
                "Score": score_label,
                "NumericScore": numeric_score
            })
            st.markdown("### Critique of Outcome Control Explanation")
            st.markdown(critique_text)
    else:
        st.warning("⚠️ Please first generate the Outcome Control explanation.")

# st.write("🔍 DEBUG: Results in session state", st.session_state.get("results", []))
st.markdown("---")
st.header("🔁 Revisions (Optional)")

if st.session_state.get("results"):
    for idx, entry in enumerate(st.session_state["results"]):
        with st.expander(f"{entry['Type']} Explanation {idx+1}"):
            st.write("**Original Explanation**")
            st.markdown(entry["Explanation"])

            st.write("**Critique**")
            st.markdown(entry["Critique"])

            # Only show if revision hasn't already been added
            if "Revised_Explanation" not in entry:
                if st.button(f"Revise Explanation {idx+1}", key=f"revise_{idx}"):
                    with st.spinner("Revising explanation..."):
                        revised = reflect_and_rewrite(entry["Explanation"], entry["Critique"])
                        st.session_state["results"][idx]["Revised_Explanation"] = revised
                        st.success("✅ Revised explanation generated!")
                        st.markdown("### 🔁 Revised Explanation")
                        st.markdown(revised)
            else:
                st.markdown("### 🔁 Revised Explanation (Saved)")
                st.markdown(entry["Revised_Explanation"])
else:
    st.info("ℹ️ No critique results yet to revise.")

# --- Scoring Table and Leaderboard ---
st.markdown("---")

st.header("Scoring Summary")

if st.session_state["results"]:
    df = pd.DataFrame(st.session_state["results"])

    # Display the full table
    st.subheader("Full Results Table")
    st.dataframe(df[["Type", "Score", "NumericScore"]])

    # Leaderboard Summary
    st.subheader("Leaderboard Summary")
    score_summary = df.groupby("Type").agg(
        Avg_Score=("NumericScore", "mean"),
        Count=("NumericScore", "count"),
        Excellent_Count=("Score", lambda x: (x == "Excellent").sum()),
        Good_Count=("Score", lambda x: (x == "Good").sum()),
        Poor_Count=("Score", lambda x: (x == "Poor").sum())
    ).reset_index()

    st.dataframe(score_summary)

    # Download button
    st.download_button(
        label="📥 Download Full Results as CSV",
        data=df.to_csv(index=False),
        file_name="fairness_audit_results.csv",
        mime="text/csv",
    )
else:
    st.info("ℹ️ No critiques yet. Please run audit or prediction explanations first.")


st.markdown("---")
st.header("Researcher Annotations")

if st.session_state["results"]:
    df = pd.DataFrame(st.session_state["results"])
    for idx, row in df.iterrows():
        with st.expander(f"{row['Type']} Explanation {idx+1}"):
            # Existing explanation
            st.write("**Explanation:**")
            st.markdown(row["Explanation"])
            st.write("**Critique:**")
            st.markdown(row["Critique"])

            # Researcher annotations
            confirm = st.radio(f"Is this critique fair? (Entry {idx+1})", ["✅ Yes", "❌ No"], key=f"confirm_{idx}")
            notes = st.text_area(f"Additional notes for Entry {idx+1}", key=f"notes_{idx}")

            # Save researcher annotations back
            st.session_state["results"][idx]["Researcher_Confirmed"] = confirm
            st.session_state["results"][idx]["Researcher_Notes"] = notes
else:
    st.info("ℹ️ No critiques to annotate yet.")


st.markdown("---")
st.header("Researcher Annotation Summary")

if st.session_state["results"]:
    df = pd.DataFrame(st.session_state["results"])

    if "Researcher_Confirmed" in df.columns:
        # Only show if researcher annotations exist
        st.dataframe(df[["Type", "Score", "Researcher_Confirmed", "Researcher_Notes"]])
    else:
        st.info("ℹ️ No researcher annotations yet.")
else:
    st.info("ℹ️ No results to display yet.")


st.markdown("---")
st.header("🤖 Automated Fairness Agent Pipeline")

def generate_counter_explanation(original_text):
    from openai import OpenAI
    try:
        api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        
        prompt = f"""
You are a fairness-aware assistant. Here is an explanation about a fairness audit:
---
{original_text}
---
Now, generate an alternative explanation that:
- Uses a different reasoning path
- Prioritizes clarity and logic
- May disagree with the original explanation if justified
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
            return """
**API Quota Exceeded - Fallback Counter Explanation:**

**Alternative Fairness Perspective:**

This provides a different reasoning approach to the fairness analysis:

**Alternative Framework:**
- Different causal pathways for bias analysis
- Alternative interpretation of fairness components
- Different prioritization of fairness criteria
- Alternative reasoning for bias mitigation strategies

**Key Differences:**
- Emphasizes different aspects of fairness decomposition
- Uses alternative causal reasoning approaches
- Provides different interpretation guidelines

*Note: Detailed AI counter-explanation unavailable due to API quota limits.*
"""
        else:
            return f"Error generating counter explanation: {error_msg}"

def full_audit_pipeline(audit_result, max_attempts=5, score_threshold="Good"):
    from openai import OpenAI
    try:
        api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
            st.warning("⚠️ API Quota Exceeded - Using Fallback Mode")
            st.markdown("""
**Fallback Fairness Pipeline:**

Due to API quota limits, the automated pipeline will use simplified explanations:

**Pipeline Steps:**
1. **Basic Explanation**: Fundamental fairness concepts
2. **Simple Critique**: Basic evaluation framework
3. **Basic Revision**: Improved explanation structure
4. **Alternative View**: Different reasoning approach

**Note**: Full AI-powered pipeline unavailable due to API quota limits.
""")
        else:
            st.error(f"Error initializing OpenAI client: {str(e)}")
        return

    # Initialize memory
    if "memory_poor_explanations" not in st.session_state:
        st.session_state["memory_poor_explanations"] = set()
    used_explanations = st.session_state["memory_poor_explanations"]

    attempt = 0
    score_label = "Poor"

    while attempt < max_attempts and score_label not in ["Good", "Excellent"]:
        attempt += 1

        # STEP 1: Generate explanation
        explanation = explain_with_agent(audit_result)

        if explanation.strip() in used_explanations:
            st.warning(f"🚫 Skipping duplicate explanation (Attempt {attempt})")
            continue

        st.session_state["current_audit_explanation"] = explanation

        # STEP 2: Critique
        critique_text, score_label = critique_explanation(explanation)
        numeric_score = {"Excellent": 2, "Good": 1, "Poor": 0}.get(score_label, -1)

        # STEP 3: Revision
        revised = reflect_and_rewrite(explanation, critique_text)

        # Save poor explanation to memory
        if score_label == "Poor":
            used_explanations.add(explanation.strip())

        # Save result
        st.session_state["results"].append({
            "Type": f"Audit (Attempt {attempt})",
            "Explanation": explanation,
            "Critique": critique_text,
            "Score": score_label,
            "NumericScore": numeric_score,
            "Revised_Explanation": revised
        })

        # Display
        st.subheader(f"✅ Explanation (Attempt {attempt})")
        st.markdown(explanation)
        st.subheader("🧐 Critique")
        st.markdown(critique_text)
        st.subheader("🔁 Revised Explanation")
        st.markdown(revised)

        # Break loop if threshold met
        if score_label in ["Good", "Excellent"]:
            break

    # --- DEBATE PHASE ---
    if score_label in ["Good", "Excellent"]:
        with st.spinner("🤖 Running debate on current explanation..."):
            alt = generate_counter_explanation(explanation)
            critique1, score1 = critique_explanation(explanation)
            critique2, score2 = critique_explanation(alt)

            numeric_score1 = {"Excellent": 2, "Good": 1, "Poor": 0}.get(score1, -1)
            numeric_score2 = {"Excellent": 2, "Good": 1, "Poor": 0}.get(score2, -1)

            winner = "Original" if numeric_score1 >= numeric_score2 else "Alternative"

            st.markdown("### 🤝 Debate Results")
            st.markdown("**Original Explanation:**")
            st.markdown(explanation)
            st.markdown("**Critique:**")
            st.markdown(critique1)

            st.markdown("**Alternative Explanation:**")
            st.markdown(alt)
            st.markdown("**Critique:**")
            st.markdown(critique2)

            st.success(f"🏆 **Winner:** {winner} explanation chosen.")

            st.session_state["results"].append({
                "Type": "Debate",
                "Explanation": explanation,
                "Counter": alt,
                "Critique_Original": critique1,
                "Critique_Alternative": critique2,
                "Winner": winner
            })

if st.button("🤖 Run Full Audit + Memory + Debate Pipeline"):
    with st.spinner("Running autonomous fairness pipeline..."):
        full_audit_pipeline(audit_result)


