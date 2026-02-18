# ml/insight_core.py

from .utils import (
    clean_data,
    validate_data,
    get_top5_text_insights,
    get_top5_visual_insights,
    create_report
)


class InsightEngine:
    """
    InsightEngine is responsible ONLY for descriptive analytics:
    - dataset summary
    - textual insights
    - visual insights
    - optional report generation

    It does NOT perform prediction or forecasting.
    """

    def __init__(self, pipeline=None):
        # Pipeline kept for future ML-based insight extensions
        self.pipeline = pipeline

    def run(self, df, generate_report=False):
        """
        Run insight generation on uploaded dataframe.

        Parameters:
        - df: pandas DataFrame
        - generate_report: bool (optional PDF report)

        Returns:
        - dict containing summary, insights, visuals
        """

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
        # 4️⃣ Visual insights
        # ---------------------------
        visual_insights = get_top5_visual_insights(df)

        # ---------------------------
        # 5️⃣ Optional report
        # ---------------------------
        report_path = None
        if generate_report:
            report_path = create_report(text_insights, visual_insights)

        # ---------------------------
        # 6️⃣ Final structured output
        # ---------------------------
        return {
            "rows": df.shape[0],
            "columns": list(df.columns),
            "missing": missing_summary,
            "text_insights": text_insights,
            "visual_insights": visual_insights,
            "report": report_path
        }
        
