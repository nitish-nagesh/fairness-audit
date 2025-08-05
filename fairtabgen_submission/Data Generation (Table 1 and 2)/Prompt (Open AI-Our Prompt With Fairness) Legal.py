from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key="your-api-key-here")

rows_per_batch = 100  # Number of rows per request
total_batches = 25    # 
output_file = "generated_data_Our_prompts_Legal.csv"

# CSV Header (comma-separated)
header = "sex, race, lsat, ugpa, zfygpa, bar_passed, tier"

# Open file and start writing
with open(output_file, "w", encoding="utf-8") as f:
    f.write(header + "\n")

    for batch in tqdm(range(total_batches), desc="Generating batches"):
        prompt = f"""
        System role: ’You are a tabular synthetic data generation model.’
        You are a synthetic data generator.
        Your goal is to produce data which mirrors \
        the given examples in causal fairness within a structural causal model (SCM) framework and feature and label distributions \
        but also produce as diverse samples as possible.
        I will give you real examples first.
        
        Context: Leverage your knowledge about Legal, LSAC National Longitudinal Bar Passage Study and causal fairness 
        to generate {rows_per_batch} realistic but diverse samples. 
        Generated data should consider 'race' (race from 1 to 8, 8 different races) as the sensitive attribute (X), 'sex' as the mediators (Z), 
        'bar_passed' as the target variable/Outcome (Y), and the rest of the features as the confounder attribute (W).
        Generated data must be structured to allow evaluation of fairness through causal pathways, capturing both direct and 
        indirect effects of the sensitive attribute on the target variable, as well as possible confounding influences.

        example data:
        sex, race, lsat, ugpa, zfygpa, bar_passed, tier
        1.0,7.0,33.0,3.5,0.58,1.0,3.0
        1.0,7.0,33.0,3.5,0.17,1.0,3.0
        1.0,7.0,31.0,3.0,0.6,1.0,3.0
        2.0,7.0,46.0,3.1,2.03,1.0,6.0
        2.0,7.0,47.0,2.8,0.29,1.0,6.0
        1.0,7.0,39.0,2.6,-0.97,0.0,3.0
        1.0,7.0,36.0,2.9,-0.08,0.0,3.0
        1.0,7.0,22.0,2.6,-1.14,0.0,3.0
        1.0,7.0,36.0,3.9,-0.63,0.0,4.0
        2.0,7.0,40.0,2.4,-0.82,0.0,3.0
        1.0,4.0,40.0,2.5,0.15,1.0,3.0
        1.0,4.0,31.5,3.6,0.83,1.0,1.0
        2.0,4.0,38.0,3.6,-1.08,1.0,5.0
        1.0,4.0,37.0,3.3,1.45,1.0,4.0
        1.0,4.0,34.0,2.9,-1.01,1.0,3.0
        2.0,4.0,28.5,2.6,-0.64,0.0,5.0
        2.0,4.0,25.5,2.8,-0.09,0.0,1.0
        2.0,4.0,25.0,3.7,-0.81,0.0,1.0
        2.0,4.0,23.0,2.5,-1.12,0.0,1.0
        2.0,4.0,32.0,3.0,-0.81,0.0,4.0
        2.0,2.0,27.0,2.9,0.29,1.0,3.0
        2.0,2.0,42.0,3.7,0.29,1.0,6.0
        2.0,2.0,26.7,3.1,-0.22,1.0,3.0
        1.0,2.0,44.0,3.2,-0.71,1.0,5.0
        2.0,2.0,23.0,3.6,-1.52,1.0,3.0
        2.0,2.0,31.0,2.6,-1.26,0.0,5.0
        2.0,2.0,26.0,2.5,-0.5,0.0,3.0
        1.0,2.0,29.0,3.5,-1.35,0.0,3.0
        1.0,2.0,29.5,3.1,-0.12,0.0,5.0
        1.0,2.0,36.5,3.4,-0.74,0.0,3.0
        1.0,3.0,30.0,2.5,-1.72,1.0,3.0
        2.0,3.0,27.0,3.7,-1.17,1.0,4.0
        1.0,3.0,13.5,2.7,-1.57,1.0,5.0
        1.0,3.0,35.0,3.1,-1.66,1.0,6.0
        1.0,3.0,35.0,2.5,-1.01,1.0,5.0
        1.0,3.0,28.0,3.4,-2.37,0.0,4.0
        2.0,3.0,26.0,2.3,-1.58,0.0,3.0
        2.0,3.0,28.0,3.1,-2.16,0.0,4.0
        2.0,3.0,26.0,2.6,-1.14,0.0,4.0
        1.0,3.0,30.0,2.9,-0.56,0.0,3.0
        2.0,6.0,31.0,2.5,-1.09,1.0,3.0
        2.0,6.0,33.0,3.3,0.96,1.0,3.0
        2.0,6.0,27.0,3.6,-0.1,1.0,3.0
        2.0,6.0,43.0,3.4,0.06,1.0,6.0
        1.0,6.0,41.0,3.0,-0.02,1.0,5.0
        2.0,6.0,39.0,2.9,-1.43,0.0,6.0
        1.0,6.0,17.5,2.5,-1.53,0.0,3.0
        1.0,6.0,27.0,2.5,-1.39,0.0,5.0
        1.0,6.0,38.0,2.4,-1.75,0.0,3.0
        2.0,6.0,22.0,2.6,-1.41,0.0,3.0
        1.0,8.0,30.5,2.7,-0.99,1.0,3.0
        1.0,8.0,36.5,3.6,-0.31,1.0,3.0
        1.0,8.0,31.5,3.6,-1.0,1.0,4.0
        2.0,8.0,38.0,3.4,1.13,1.0,3.0
        1.0,8.0,36.0,3.5,0.76,1.0,3.0
        2.0,8.0,36.0,3.5,-0.19,0.0,3.0
        1.0,8.0,32.0,2.7,-0.79,0.0,4.0
        1.0,8.0,39.0,2.8,-0.97,0.0,5.0
        2.0,8.0,36.0,3.3,-1.55,0.0,3.0
        1.0,8.0,40.0,3.1,-1.31,0.0,4.0
        1.0,1.0,28.0,3.9,-0.01,1.0,4.0
        2.0,1.0,26.0,3.5,-0.68,1.0,2.0
        1.0,1.0,39.5,2.9,-1.11,1.0,5.0
        1.0,1.0,32.0,3.9,0.77,1.0,3.0
        2.0,1.0,30.5,3.6,-1.29,1.0,6.0
        2.0,1.0,25.0,3.4,-1.77,0.0,3.0
        2.0,1.0,26.0,3.4,-0.99,0.0,4.0
        1.0,1.0,29.0,2.9,-1.36,0.0,4.0
        1.0,1.0,26.0,2.1,-1.62,0.0,4.0
        1.0,1.0,40.0,2.5,-0.94,0.0,2.0
        2.0,5.0,40.0,3.0,0.25,1.0,5.0
        1.0,5.0,36.0,3.5,0.57,1.0,3.0
        2.0,5.0,41.0,3.2,-1.17,1.0,4.0
        2.0,5.0,35.0,3.2,-0.1,1.0,3.0
        1.0,5.0,27.5,3.3,-0.48,1.0,5.0
        2.0,5.0,21.0,3.4,-0.27,0.0,3.0
        2.0,5.0,31.0,3.0,-1.46,0.0,3.0
        2.0,5.0,28.0,2.8,-0.42,0.0,1.0
        1.0,5.0,26.0,3.4,-1.51,0.0,4.0
        2.0,5.0,29.0,2.6,-2.05,0.0,5.0

        DO NOT COPY THE EXAMPLES but generate realistic but new AND diverse samples which have
        the correct label conditioned on the features.

        """

        # Stream the response
        buffer=""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            temperature=0.9,
            top_p=0.95,
            max_tokens=8*2048
        )

        # Process stream and write to file
        for chunk in response:
            if chunk.choices[0].delta.content:
                buffer += chunk.choices[0].delta.content
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line and line.count(",") == 6:  # Exactly 7 fields
                        f.write(line.replace(",", ";") + "\n")