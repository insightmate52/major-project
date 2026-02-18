import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os

# ==================================================
# CONFIG
# ==================================================

INSIGHT_DIR = "static/insights"
os.makedirs(INSIGHT_DIR, exist_ok=True)

# ==================================================
# UTIL: SAVE PLOT
# ==================================================

def save_plot(fig, filename):
    path = os.path.join(INSIGHT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return filename

# ==================================================
# RED-FLAG DETECTORS
# ==================================================

def detect_outliers(df):
    flags = []
    num_cols = df.select_dtypes(include="number").columns

    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_ratio = (
            (df[col] < q1 - 1.5 * iqr) |
            (df[col] > q3 + 1.5 * iqr)
        ).mean()

        if outlier_ratio > 0.05:
            flags.append({
                "type": "outlier",
                "column": col,
                "severity": round(outlier_ratio, 2)
            })

    return flags


def detect_declining_trend(df):
    flags = []
    num_cols = df.select_dtypes(include="number").columns

    if len(df) < 10:
        return flags

    for col in num_cols:
        slope = np.polyfit(range(len(df)), df[col].values, 1)[0]
        if slope < 0:
            flags.append({
                "type": "decline",
                "column": col,
                "severity": round(abs(slope), 3)
            })

    return flags


def detect_volatility(df):
    flags = []
    num_cols = df.select_dtypes(include="number").columns

    for col in num_cols:
        mean = df[col].mean()
        std = df[col].std()
        if mean == 0:
            continue

        cv = std / mean
        if cv > 0.5:
            flags.append({
                "type": "volatility",
                "column": col,
                "severity": round(cv, 2)
            })

    return flags


def detect_dangerous_correlation(df):
    flags = []
    num_df = df.select_dtypes(include="number")

    if num_df.shape[1] < 2:
        return flags

    corr = num_df.corr()

    for i in corr.columns:
        for j in corr.columns:
            if i != j and abs(corr.loc[i, j]) > 0.85:
                flags.append({
                    "type": "correlation",
                    "columns": (i, j),
                    "severity": round(abs(corr.loc[i, j]), 2)
                })

    return flags


def detect_category_dominance(df):
    flags = []
    cat_cols = df.select_dtypes(exclude="number").columns

    for col in cat_cols:
        vc = df[col].dropna().value_counts(normalize=True)

        if vc.empty:
            continue   # ✅ avoid out-of-bounds

        top_ratio = vc.iloc[0]

        if top_ratio > 0.7:
            flags.append({
                "type": "category_risk",
                "column": col,
                "severity": round(top_ratio, 2)
            })

    return flags

# ==================================================
# GRAPH + EXPLANATION GENERATOR
# ==================================================

def generate_visual(flag, df):
    fid = str(uuid.uuid4())[:8]

    # ---------- OUTLIERS ----------
    if flag["type"] == "outlier":
        col = flag["column"]
        fig = plt.figure()
        sns.boxplot(x=df[col])
        img = save_plot(fig, f"outlier_{fid}.png")

        explanation = (
            f"Significant outliers were detected in '{col}'. "
            f"This indicates abnormal values that may represent data errors, "
            f"fraudulent activity, or exceptional cases requiring immediate review."
        )

    # ---------- DECLINING TREND ----------
    elif flag["type"] == "decline":
        col = flag["column"]
        fig = plt.figure()
        plt.plot(df[col])
        plt.title(f"Declining Trend in {col}")
        img = save_plot(fig, f"decline_{fid}.png")

        explanation = (
            f"The metric '{col}' shows a consistent downward trend. "
            f"This is a critical red flag indicating performance deterioration "
            f"and may require urgent corrective action."
        )

    # ---------- VOLATILITY ----------
    elif flag["type"] == "volatility":
        col = flag["column"]
        fig = plt.figure()
        sns.histplot(df[col], kde=True)
        img = save_plot(fig, f"volatility_{fid}.png")

        explanation = (
            f"High volatility detected in '{col}', indicating unstable behavior. "
            f"Such unpredictability increases risk and makes forecasting difficult."
        )

    # ---------- CORRELATION ----------
    elif flag["type"] == "correlation":
        fig = plt.figure(figsize=(6, 4))
        sns.heatmap(
            df.select_dtypes(include="number").corr(),
            annot=True,
            cmap="coolwarm"
        )
        img = save_plot(fig, f"correlation_{fid}.png")

        explanation = (
            "Strong correlation detected between numerical variables. "
            "This suggests dependency where changes in one metric may "
            "significantly impact others, potentially amplifying risk."
        )

    # ---------- CATEGORY DOMINANCE ----------
    else:
        col = flag["column"]
        fig = plt.figure()
        df[col].value_counts().plot(kind="bar")
        img = save_plot(fig, f"category_{fid}.png")

        explanation = (
            f"The category '{col}' is highly imbalanced, with one group dominating. "
            f"This creates dependency risk and reduces diversification."
        )

    return {
        "image": img,
        "explanation": explanation,
        "severity": flag["severity"]
    }

# ==================================================
# MAIN ENTRY POINT
# ==================================================

def generate_top_red_flags(df, top_n=5):
    flags = []
    flags += detect_outliers(df)
    flags += detect_declining_trend(df)
    flags += detect_volatility(df)
    flags += detect_dangerous_correlation(df)
    flags += detect_category_dominance(df)

    if not flags:
        return []

    flags = sorted(flags, key=lambda x: x["severity"], reverse=True)[:top_n]

    visuals = [generate_visual(flag, df) for flag in flags]
    return visuals