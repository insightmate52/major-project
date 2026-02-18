# ml/insight_engine.py

from ml.utils import (
    clean_data,
    validate_data,
    get_top5_text_insights,
    get_top5_visual_insights,
    create_report
)


class InsightEngine:
    """
    Generates dataset summary, textual insights,
    and visual insights with explanations.
    """

    def __init__(self, pipeline=None):
        # Pipeline kept for future ML-based insight extension
        self.pipeline = pipeline

    def run(self, df):
        # ---------------------------
        # 1️⃣ Clean & validate data
        # ---------------------------
        df = clean_data(df)
        validate_data(df)

        # ---------------------------
        # 2️⃣ Dataset summary
        # ---------------------------
        missing_summary = (
            df.isnull().sum()
            .loc[lambda x: x > 0]
            .to_dict()
        )

        # ---------------------------
        # 3️⃣ Textual insights
        # ---------------------------
        text_insights = get_top5_text_insights(df)

        # ---------------------------
        # 4️⃣ Visual insights + explanation
        # ---------------------------
        visual_insights = get_top5_visual_insights(df)

        visual_explanations = [
            "This chart highlights an important pattern or distribution in the data."
            for _ in visual_insights
        ]

        # ---------------------------
        # 5️⃣ Optional report
        # ---------------------------
        report_path = create_report(text_insights, visual_insights)

        return {
            "rows": df.shape[0],
            "columns": list(df.columns),
            "missing": missing_summary,
            "text_insights": text_insights,
            "visual_insights": visual_insights,
            "visual_explanations": visual_explanations,
            "report": report_path
        }
