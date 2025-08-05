from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key="your-api-key-here")

rows_per_batch = 100  # Number of rows per request
total_batches = 20    # 20 batches × 100 rows = 1000 rows
output_file = "generated_data_CLLM_prompt_Criminal Justice.csv"

# CSV Header (comma-separated)
header = "sex,age,juv_fel_count,juv_misd_count,juv_other_count,priors_count,two_year_recid,age_cat_25-45,age_cat_Greaterthan45,age_cat_Lessthan25,race_African-American,race_Caucasian,c_charge_degree_F,c_charge_degree_M"

# Open file and start writing
with open(output_file, "w", encoding="utf-8") as f:
    f.write(header + "\n")

    for batch in tqdm(range(total_batches), desc="Generating batches"):
        prompt = f"""
        System role: ’You are a tabular synthetic data generation model.’
        You are a synthetic data generator.
        Your goal is to produce data which mirrors \
        the given examples in causal structure and feature and label distributions \
        but also produce as diverse samples as possible.

        I will give you real examples first.

        Context: Leverage your prior knowledge to generate {rows_per_batch}
        realistic but diverse samples.

        example data:
        sex,age,juv_fel_count,juv_misd_count,juv_other_count,priors_count,two_year_recid,age_cat_25-45,age_cat_Greaterthan45,age_cat_Lessthan25,race_African-American,race_Caucasian,c_charge_degree_F,c_charge_degree_M
        1,24,0,0,0,2,1,0,0,1,1,0,1,0
        1,20,1,1,1,4,1,0,0,1,1,0,1,0
        1,25,0,0,1,1,1,1,0,0,1,0,0,1
        0,31,0,0,0,0,1,1,0,0,0,1,1,0
        1,25,0,0,0,0,1,1,0,0,1,0,0,1
        1,41,0,0,0,0,1,1,0,0,0,1,1,0
        1,25,0,0,0,0,0,1,0,0,0,1,1,0
        1,25,0,0,0,2,1,1,0,0,1,0,1,0
        1,22,0,0,0,1,1,0,0,1,1,0,1,0
        0,25,0,0,0,0,0,1,0,0,1,0,1,0
        1,38,0,0,0,0,0,1,0,0,1,0,0,1
        0,49,0,0,0,4,0,0,1,0,0,1,0,1
        0,35,0,0,0,2,0,1,0,0,0,1,0,1
        1,21,0,0,0,1,0,0,0,1,0,1,1,0
        0,24,0,0,1,11,1,0,0,1,1,0,0,1
        1,28,0,0,0,3,0,1,0,0,0,1,1,0
        1,34,0,0,0,4,1,1,0,0,1,0,1,0
        1,57,0,0,0,8,0,0,1,0,1,0,0,1
        1,40,0,0,0,15,1,1,0,0,1,0,1,0
        1,22,0,0,0,2,1,0,0,1,1,0,0,1
        1,37,0,0,0,3,1,1,0,0,1,0,0,1
        0,43,0,0,0,0,0,1,0,0,1,0,1,0
        1,26,0,0,0,5,1,1,0,0,1,0,1,0
        1,24,0,0,0,2,1,0,0,1,0,1,1,0
        1,56,0,0,0,1,1,0,1,0,0,1,1,0
        1,23,0,0,1,2,1,0,0,1,1,0,1,0
        1,43,0,0,0,12,0,1,0,0,0,1,1,0
        1,21,0,0,0,0,1,0,0,1,1,0,1,0
        1,51,0,0,0,2,0,0,1,0,1,0,1,0
        1,20,0,0,0,0,1,0,0,1,1,0,0,1
        1,42,0,0,0,17,1,1,0,0,1,0,1,0
        0,30,0,0,0,2,0,1,0,0,1,0,0,1
        1,28,1,1,1,4,1,1,0,0,0,1,0,1
        0,24,0,0,0,9,1,0,0,1,0,1,0,1
        1,22,0,0,0,1,1,0,0,1,1,0,1,0
        1,50,0,0,0,1,0,0,1,0,0,1,1,0
        1,30,0,0,0,5,0,1,0,0,1,0,0,1
        1,28,0,0,0,0,1,1,0,0,0,1,1,0
        1,21,0,0,0,3,1,0,0,1,1,0,1,0
        0,51,0,0,0,0,0,0,1,0,0,1,1,0

        The output should be a markdown code snippet formatted in the following schema:
        sex,age,juv_fel_count,juv_misd_count,juv_other_count,priors_count,two_year_recid,age_cat_25-45,age_cat_Greaterthan45,age_cat_Lessthan25,race_African-American,race_Caucasian,c_charge_degree_F,c_charge_degree_M
        
        DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have
        the correct label conditioned on the features.

        """

        # Stream the response
        buffer=""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                 {"role": "user", "content": prompt}
            ],
            stream=True,
            temperature=0.9,
            top_p=1.0,
            max_tokens=8*2048
        )

        # Process stream and write to file
        for chunk in response:
            if chunk.choices[0].delta.content:
                buffer += chunk.choices[0].delta.content
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line and line.count(",") == 13:  # Exactly 14 fields
                        f.write(line.replace(",", ";") + "\n")