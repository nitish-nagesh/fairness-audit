
import os
import re
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro

# --- Config ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_CSV = "refined_fairgen_data.csv"
HEADER = "sex,age,juv_fel,juv_misd,juv_other,priors,charge,two_year_recid,race"
TARGET_MEAN = 0.01
TARGET_SD = 0.005
MAX_ITER = 10
ROWS_PER_BATCH = 100

example_block = """sex,age,juv_fel,juv_misd,juv_other,priors,charge,two_year_recid,race
1,69,0,0,0,0,1,0,0
1,34,0,0,0,0,1,1,0
1,24,0,0,1,4,1,1,0
1,23,0,1,0,1,1,0,0
1,43,0,0,0,2,1,0,0
1,44,0,0,0,0,0,0,0
1,41,0,0,0,14,1,1,1
1,43,0,0,0,3,1,0,0
0,39,0,0,0,0,0,0,1
1,21,0,0,0,1,1,1,1"""

def generate_prompt(refinement_notes=""):
    return f"""
System role: 'You are a tabular synthetic data generation model.'
You are a synthetic data generator.

Your goal is to produce data which mirrors the examples in causal fairness within a structural causal model (SCM) framework and feature and label distributions but also produce as diverse samples as possible.

Context: Leverage your knowledge about criminal recidivism and causal fairness to generate {ROWS_PER_BATCH} realistic but diverse samples.

Generated data should consider:
- Sensitive attribute (X): race (0 = Minority, 1 = Majority)
- Mediators (Z): age, sex
- Outcome (Y): two_year_recid
- Confounders (W): juv_fel, juv_misd, juv_other, priors, charge

**Generated samples must reduce**:
- Total Variation (tv)
- Conditional Treatment-Free Direct Effect (ctfde)
- Indirect Effect (ctfie)
- Sequential Effect (ctfse)
- Treatment Effect (ett)

Target thresholds: mean < 0.01, SD < 0.005.

{refinement_notes}

Example data:
{example_block}

DO NOT COPY the examples. Generate new but realistic rows that preserve fairness and statistical distribution.
"""

def generate_synthetic_rows(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=True,
        temperature=0.9,
        top_p=0.95,
        max_tokens=8 * 2048,
    )
    buffer, rows = "", []
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            buffer += delta.content
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if re.match(r"^[01],[0-9.]+,\d+,\d+,\d+,\d+,[01],\d+,[01]$", line):
                    rows.append(line)
    return rows

def run_r_fairness_audit(csv_path):
    faircause = importr("faircause")
    df = pd.read_csv(csv_path)
    required_cols = set(["two_year_recid", "sex", "race", "age", "juv_fel", "juv_misd", "juv_other", "priors", "charge"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)
    fc = faircause.fairness_cookbook(
        r_df, X="race",
        W=StrVector(["juv_fel", "juv_misd", "juv_other", "priors", "charge"]),
        Z=StrVector(["age", "sex"]),
        Y="two_year_recid",
        x0=0, x1=1
    )
    return pd.DataFrame(faircause.summary(fc).rx2("measures"))

# --- Auto-Refinement Loop ---
for iteration in range(1, MAX_ITER + 1):
    print(f"\n🔁 Iteration {iteration}")
    prompt = generate_prompt(f"Refinement step {iteration}. Try reducing all fairness metrics.")
    synthetic_rows = generate_synthetic_rows(prompt)

    with open(OUTPUT_CSV, "w") as f:
        f.write(HEADER + "\n")
        for row in synthetic_rows:
            f.write(row + "\n")

    try:
        fairness_df = run_r_fairness_audit(OUTPUT_CSV)
        filtered = fairness_df[fairness_df["measure"].isin(["tv", "ctfde", "ctfie", "ctfse", "ett"])]
        max_mean = filtered["value"].abs().max()
        max_sd = filtered["sd"].max()
        print(filtered[["measure", "value", "sd"]])
        print(f"📊 Max |mean|: {max_mean:.4f}, Max SD: {max_sd:.4f}")
        if max_mean < TARGET_MEAN and max_sd < TARGET_SD:
            print("✅ Thresholds met! Stopping.")
            break
    except Exception as e:
        print(f"❌ Error during audit: {e}")

    time.sleep(2)
