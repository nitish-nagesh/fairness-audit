import os
import base64
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

client = OpenAI(api_key=api_key)

rows_per_batch = 100
total_batches = 20
output_file = "fairgencompas01.csv"

# CSV header
header = "sex,age,juv_fel_count,juv_misd_count,juv_other_count,priors_count,two_year_recid,age_cat_25-45,age_cat_Greaterthan45,age_cat_Lessthan25,race_African-American,race_Caucasian,c_charge_degree_F,c_charge_degree_M"

# Example data block (use exactly as you posted)
example_block = """
sex,age,juv_fel_count,juv_misd_count,juv_other_count,priors_count,two_year_recid,age_cat_25-45,age_cat_Greaterthan45,age_cat_Lessthan25,race_African-American,race_Caucasian,c_charge_degree_F,c_charge_degree_M
1,31.0,0.0,0.0,0.0,2.0,1,1,0,0,1,0,1.0,0.0
1,44.0,0.0,0.0,0.0,6.0,1,1,0,0,1,0,1.0,0.0
1,35.0,0.0,1.0,0.0,14.0,1,1,0,0,1,0,0.0,1.0
1,43.0,0.0,0.0,0.0,9.0,1,1,0,0,1,0,1.0,0.0
1,22.0,0.0,1.0,0.0,2.0,1,0,0,1,1,0,1.0,0.0
1,26.0,0.0,0.0,0.0,8.0,1,1,0,0,1,0,1.0,0.0
1,33.0,0.0,0.0,0.0,1.0,1,1,0,0,1,0,0.0,1.0
1,43.0,0.0,0.0,0.0,3.0,1,1,0,0,1,0,0.0,1.0
1,24.0,0.0,0.0,0.0,17.0,1,0,0,1,1,0,1.0,0.0
1,27.0,0.0,0.0,0.0,6.0,1,1,0,0,1,0,1.0,0.0
0,52.0,0.0,0.0,0.0,0.0,0,0,1,0,1,0,1.0,0.0
1,63.0,0.0,0.0,0.0,0.0,0,0,1,0,1,0,0.0,1.0
1,31.0,0.0,0.0,0.0,0.0,0,1,0,0,1,0,1.0,0.0
1,23.0,0.0,0.0,0.0,0.0,0,0,0,1,1,0,1.0,0.0
1,31.0,0.0,0.0,0.0,5.0,0,1,0,0,1,0,0.0,1.0
1,25.0,0.0,0.0,0.0,0.0,0,1,0,0,1,0,1.0,0.0
1,20.0,0.0,0.0,0.0,1.0,0,0,0,1,1,0,0.0,1.0
1,40.0,0.0,0.0,0.0,0.0,0,1,0,0,1,0,1.0,0.0
1,22.0,0.0,0.0,0.0,0.0,0,0,0,1,1,0,0.0,1.0
1,37.0,0.0,0.0,0.0,17.0,0,1,0,0,1,0,0.0,1.0
1,31.0,0.0,0.0,3.0,9.0,1,1,0,0,0,1,1.0,0.0
1,43.0,0.0,0.0,0.0,11.0,1,1,0,0,0,1,1.0,0.0
1,27.0,0.0,0.0,0.0,1.0,1,1,0,0,0,1,0.0,1.0
1,22.0,0.0,0.0,0.0,0.0,1,0,0,1,0,1,1.0,0.0
0,31.0,0.0,0.0,0.0,1.0,1,1,0,0,0,1,0.0,1.0
1,35.0,0.0,0.0,0.0,2.0,1,1,0,0,0,1,0.0,1.0
1,25.0,0.0,0.0,0.0,3.0,1,1,0,0,0,1,0.0,1.0
1,38.0,0.0,0.0,0.0,1.0,1,1,0,0,0,1,1.0,0.0
1,25.0,0.0,0.0,0.0,5.0,1,1,0,0,0,1,1.0,0.0
1,23.0,0.0,0.0,1.0,4.0,1,0,0,1,0,1,1.0,0.0
1,26.0,0.0,0.0,0.0,1.0,0,1,0,0,0,1,0.0,1.0
1,56.0,0.0,0.0,0.0,13.0,0,0,1,0,0,1,1.0,0.0
1,33.0,0.0,0.0,0.0,1.0,0,1,0,0,0,1,1.0,0.0
1,28.0,0.0,0.0,0.0,0.0,0,1,0,0,0,1,1.0,0.0
1,37.0,0.0,0.0,0.0,0.0,0,1,0,0,0,1,1.0,0.0
1,33.0,0.0,0.0,0.0,2.0,0,1,0,0,0,1,1.0,0.0
1,26.0,0.0,0.0,0.0,0.0,0,1,0,0,0,1,1.0,0.0
1,30.0,0.0,0.0,0.0,0.0,0,1,0,0,0,1,1.0,0.0
0,33.0,0.0,0.0,0.0,7.0,0,1,0,0,0,1,0.0,1.0
1,22.0,0.0,0.0,1.0,0.0,0,0,0,1,0,1,1.0,0.0
        

"""  # Truncated for clarity—you can paste your full example block here

# Main generation loop
with open(output_file, "w", encoding="utf-8") as f:
    f.write(header + "\n")

    for batch in tqdm(range(total_batches), desc="Generating synthetic batches"):
        prompt = f"""
        System role: ’You are a tabular synthetic data generation model.’
        You are a synthetic data generator.
        Your goal is to produce data which mirrors the given examples in causal fairness within a structural causal model (SCM) framework and feature and label distributions but also produce as diverse samples as possible.
        I will give you real examples first.

        Context: Leverage your knowledge about criminal recidivism and causal fairness to generate {rows_per_batch} realistic but diverse samples. Generated data should consider 'race' ('race_African-American'/'race_Caucasian') as the sensitive attribute (X), 'age' and 'sex' as the mediators (Z), 'two_year_recid' as the target variable/Outcome (Y), and the rest of the features as the confounder attribute (W).
        Generated data must be structured to allow evaluation of fairness through causal pathways, capturing both direct and indirect effects of the sensitive attribute on the target variable, as well as possible confounding influences.

        example data:
        {example_block}

        DO NOT COPY THE EXAMPLES but generate realistic but new AND diverse samples which have the correct label conditioned on the features.
        """

        buffer = ""
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

        for chunk in response:
            if chunk.choices[0].delta.content:
                buffer += chunk.choices[0].delta.content
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line and line.count(",") == 13:  # 14 columns
                        f.write(line + "\n")

print(f"\n✅ Finished generating {rows_per_batch * total_batches} rows to {output_file}")