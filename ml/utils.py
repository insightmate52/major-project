import os
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import matplotlib.pyplot as plt
import seaborn as sns

# Lavender / Purple theme
LAVENDER_PALETTE = [
    "#8458B3",  # primary purple
    "#D0BDF4",  # soft lavender
    "#A0D2EB",  # light blue
    "#494D5F",  # dark gray
    "#E5EAF5"   # light background
]

sns.set_theme(style="whitegrid", palette=LAVENDER_PALETTE)
plt.rcParams.update({
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "axes.edgecolor": "#8458B3",
    "axes.labelcolor": "#494D5F",
    "xtick.color": "#494D5F",
    "ytick.color": "#494D5F",
    "grid.color": "#E5EAF5",
})
# ===========================
# 🧹 CLEAN DATA
# ===========================
def clean_data(df):
    df = df.reset_index(drop=True).copy()

    # -----------------------------
    # HANDLE DATE (SORT THEN DROP)
    # -----------------------------
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
        df = df.drop(columns=[date_col])   # 🔥 FINAL KILL

    # -----------------------------
    # SAFE CLEANING
    # -----------------------------
    df = df.dropna(how="all")
    df = df.ffill().bfill()

    return df

# ===========================
# 🔍 VALIDATE DATA
# ===========================
def validate_data(df: pd.DataFrame):
    if df.shape[0] < 5:
        print("⚠️ Warning: Dataset has very few rows")

    if df.shape[1] < 2:
        print("⚠️ Warning: Dataset has very few columns")


# ===========================
# 📝 TEXT INSIGHTS
# ===========================
def get_top5_text_insights(df: pd.DataFrame):
    insights = []

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if not num_cols:
        return ["Dataset does not contain numeric columns for analysis."]

    # 1️⃣ Average
    col = num_cols[0]
    insights.append(f"Average value of {col} is {df[col].mean():.2f}.")

    # 2️⃣ Correlation (SAFE FIX)
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs().copy()

        mask = np.tril(np.ones(corr.shape), k=0).astype(bool)
        corr.where(~mask, inplace=True)

        if not corr.stack().empty:
            pair = corr.stack().idxmax()
            insights.append(
                f"{pair[0]} shows strong correlation with {pair[1]}."
            )

    # 3️⃣ Category contribution
    if cat_cols:
        top_cat = df.groupby(cat_cols[0])[col].sum().idxmax()
        insights.append(f"{top_cat} contributes the highest value to {col}.")

    # 4️⃣ High value segment
    insights.append(f"Top 10% of {col} represent high-value records.")

    # 5️⃣ Risk segment
    insights.append(f"Bottom 10% of {col} may indicate potential risk.")

    return insights[:5]


# ===========================
# 📊 VISUAL INSIGHTS
# ===========================
def generate_top_red_flags(df, top_n=5):

    # 🔥 REMOVE DATE COLUMNS COMPLETELY
    df = df.loc[:, ~df.columns.str.contains("date", case=False)].copy()
    
def get_top5_visual_insights(df: pd.DataFrame, output_dir="static/insights"):
    os.makedirs(output_dir, exist_ok=True)
    images = []

    uid = uuid.uuid4().hex[:6]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if not num_cols:
        return images

    # 1️⃣ Distribution
    plt.figure()
    plt.hist(df[num_cols[0]], bins=20)
    path = f"viz_{uid}_distribution.png"
    plt.savefig(os.path.join(output_dir, path))
    plt.close()
    images.append(path)

    # 2️⃣ Scatter
    if len(num_cols) >= 2:
        plt.figure()
        plt.scatter(df[num_cols[0]], df[num_cols[1]])
        path = f"viz_{uid}_scatter.png"
        plt.savefig(os.path.join(output_dir, path))
        plt.close()
        images.append(path)

    # 3️⃣ Category bar
    if cat_cols:
        plt.figure()
        df.groupby(cat_cols[0])[num_cols[0]].sum().head(5).plot(kind="bar")
        path = f"viz_{uid}_category.png"
        plt.savefig(os.path.join(output_dir, path))
        plt.close()
        images.append(path)

    # 4️⃣ Boxplot
    plt.figure()
    plt.boxplot(df[num_cols[0]])
    path = f"viz_{uid}_boxplot.png"
    plt.savefig(os.path.join(output_dir, path))
    plt.close()
    images.append(path)

    # 5️⃣ Correlation heatmap (matplotlib only)
    if len(num_cols) >= 3:
        plt.figure(figsize=(6, 4))
        corr = df[num_cols].corr()
        plt.imshow(corr, cmap="coolwarm")
        plt.colorbar()
        plt.xticks(range(len(num_cols)), num_cols, rotation=45)
        plt.yticks(range(len(num_cols)), num_cols)
        path = f"viz_{uid}_heatmap.png"
        plt.savefig(os.path.join(output_dir, path))
        plt.close()
        images.append(path)

    return images[:5]


# ===========================
# 📄 PDF REPORT
# ===========================
def create_report(text_insights, visual_images, output_dir="static"):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(
        output_dir, f"report_{uuid.uuid4().hex[:6]}.pdf"
    )

    c = canvas.Canvas(report_path, pagesize=letter)

    c.drawString(50, 750, "Top 5 Data Insights")
    y = 720
    for i, ins in enumerate(text_insights, 1):
        c.drawString(50, y, f"{i}. {ins}")
        y -= 18

    c.showPage()

    for img in visual_images:
        img_path = os.path.join("static/insights", img)
        if os.path.exists(img_path):
            c.drawImage(img_path, 50, 300, width=500, height=350)
            c.showPage()

    c.save()
    return report_path

