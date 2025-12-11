"""
Utility script to reproduce `finalDataset.csv` from the two raw CSVs stored
under `Datasets/`.  It performs the exact cleaning/merging workflow described
in the README: standardises column names, harmonises education levels, infers
seniority, adds PPP adjustments, and tags potential fake job posts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "Datasets"
OUTPUT_PATH = PROJECT_ROOT / "finalDataset.csv"

EDU_CODE_TO_TEXT = {
    0: "Unknown",
    1: "Bachelors",
    2: "Masters",
    3: "PhD",
}

EDU_TEXT_TO_CODE = {
    "bachelors": 1,
    "bachelor's": 1,
    "bachelor's degree": 1,
    "masters": 2,
    "master's": 2,
    "master's degree": 2,
    "phd": 3,
    "doctorate": 3,
    "high school": 0,
    "associate": 0,
    "unknown": 0,
}

PPP_LOOKUP: Dict[str, float] = {
    "USA": 1.00,
    "United States": 1.00,
    "UK": 1.20,
    "United Kingdom": 1.20,
    "Canada": 1.10,
    "Australia": 1.05,
    "China": 1.00,
    # Default fallback will be applied later.
}

SENIORITY_KEYWORDS = ("senior", "lead", "manager", "principal", "director", "head", "vp", "chief")
FAKE_JOB_KEYWORDS = ("senior", "director", "vp", "chief")
FAKE_JOB_SALARY_THRESHOLD = 140_000


def normalise_job_title(title: str) -> str:
    return title.strip()


def infer_seniority(title: str) -> int:
    lowered = title.lower()
    return int(any(keyword in lowered for keyword in SENIORITY_KEYWORDS))


def map_ppp(country: str) -> float:
    return PPP_LOOKUP.get(country, 1.0)


def compute_fake_job_flag(row: pd.Series) -> int:
    high_title = any(keyword in str(row["Job Title"]).lower() for keyword in FAKE_JOB_KEYWORDS)
    return int(high_title and row["Salary_PPP_Adjusted"] >= FAKE_JOB_SALARY_THRESHOLD)


def clean_salary_numeric() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "Salary.csv")
    df = df.rename(
        columns={
            "Education Level": "Education_Level_Code",
            "Job Title": "Job Title",
        }
    )
    df["Education_Level_Code"] = df["Education_Level_Code"].astype(float)
    df["Education_Level"] = df["Education_Level_Code"].map(EDU_CODE_TO_TEXT).fillna("Unknown")
    df["Job Title"] = df["Job Title"].astype(str).map(normalise_job_title)
    df["Senior"] = df["Senior"].astype(float)
    return df[
        [
            "Age",
            "Gender",
            "Education_Level_Code",
            "Education_Level",
            "Job Title",
            "Years of Experience",
            "Salary",
            "Country",
            "Race",
            "Senior",
        ]
    ]


def clean_salary_country() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "Salary_Data_Based_country_and_race.csv")
    df = df.drop(columns=[col for col in df.columns if col.lower().startswith("unnamed")], errors="ignore")
    df = df.rename(columns={"Education Level": "Education_Level", "Years of Experience": "Years of Experience"})
    df["Education_Level"] = (
        df["Education_Level"]
        .astype(str)
        .str.strip()
        .replace("nan", "Unknown")
    )
    df["Education_Level_Code"] = (
        df["Education_Level"].str.lower().map(EDU_TEXT_TO_CODE).fillna(0).astype(float)
    )
    df["Job Title"] = df["Job Title"].astype(str).map(normalise_job_title)
    df["Senior"] = df["Job Title"].apply(infer_seniority).astype(float)
    return df[
        [
            "Age",
            "Gender",
            "Education_Level_Code",
            "Education_Level",
            "Job Title",
            "Years of Experience",
            "Salary",
            "Country",
            "Race",
            "Senior",
        ]
    ]


def build_final_dataset() -> pd.DataFrame:
    salary_numeric = clean_salary_numeric()
    salary_country = clean_salary_country()

    combined = pd.concat([salary_numeric, salary_country], ignore_index=True)
    combined["PPP_Index"] = combined["Country"].map(map_ppp).astype(float)
    combined["Salary_PPP_Adjusted"] = combined["Salary"] / combined["PPP_Index"].replace({0: np.nan})
    combined["Fake_Job_Risk"] = combined.apply(compute_fake_job_flag, axis=1)

    combined = combined.drop_duplicates()
    combined = combined.dropna(subset=["Age", "Gender", "Job Title", "Salary"])

    column_order = [
        "Age",
        "Gender",
        "Education_Level_Code",
        "Education_Level",
        "Job Title",
        "Years of Experience",
        "Salary",
        "Country",
        "Race",
        "Senior",
        "PPP_Index",
        "Salary_PPP_Adjusted",
        "Fake_Job_Risk",
    ]
    return combined[column_order]


def main() -> None:
    final_df = build_final_dataset()
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(final_df):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

