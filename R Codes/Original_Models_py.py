import subprocess
import os

# All R scripts in the current directory
r_scripts = [
    "Original_COMPAS.R",
    "Original_Models.R",
    "No_Fairness_Results.R",
    "Our_Prompt_Results.R",
    "With_Fairness_Results.R",
    "Zero_Prior_Results.R"
]

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Run each R script
for script in r_scripts:
    script_path = os.path.join(current_dir, script)
    print(f"Running: {script}")
    try:
        subprocess.run(["Rscript", script_path], check=True)
        print(f"✅ Completed: {script}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {script}\n{e}")
