### 2. fairness_audit_runner.py (External RPy2 Runner)

import sys
import pyreadr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr

def run_fairness_audit(file_path):
    result = pyreadr.read_r(file_path)
    compas = result[list(result.keys())[0]]

    faircause = importr("faircause")

    compas["race"] = compas["race"].apply(lambda r: "Majority" if r in ["Caucasian", "White"] else "Minority")
    compas.rename(columns={
        "juv_fel_count": "juv_fel",
        "juv_misd_count": "juv_misd",
        "juv_other_count": "juv_other",
        "priors_count": "priors",
        "c_charge_degree": "charge"
    }, inplace=True)

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_compas = ro.conversion.py2rpy(compas)

    result = faircause.fairness_cookbook(
        data=r_compas,
        X="race",
        W=StrVector(["juv_fel", "juv_misd", "juv_other", "priors", "charge"]),
        Z=StrVector(["age", "sex"]),
        Y="two_year_recid",
        x0="Minority",
        x1="Majority"
    )

    summary = ro.r("summary")(result)
    measures = summary.rx2("measures")

    with localconverter(ro.default_converter + pandas2ri.converter):
        df = ro.conversion.rpy2py(measures)

    output = []
    for row in df.itertuples(index=False):
        output.append(f"{row.measure}: {row.value:.4f} (±{row.sd:.4f})")

    return "\n".join(output)

if __name__ == "__main__":
    file_path = sys.argv[1]
    output = run_fairness_audit(file_path)
    with open("audit_output.txt", "w") as f:
        f.write(output)
