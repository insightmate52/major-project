def build_planning_prompt(schema_info, user_question):
    return f"""
You are a professional data analyst.

Dataset columns:
{schema_info['columns']}

If prediction is required:
Use LinearRegression from sklearn.

User Question:
{user_question}

Return ONLY valid pandas or sklearn code.
Final output must be stored in variable named result.
"""