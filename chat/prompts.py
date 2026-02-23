def build_planning_prompt(schema_info, user_question):
    return f"""
You are an expert Python data analyst.

You are working with a pandas DataFrame named df.

Available dataset columns (use EXACT names, case-sensitive):
{schema_info['columns']}

User Question:
{user_question}

STRICT RULES:
1. Use ONLY pandas or sklearn (LinearRegression if prediction is required).
2. Use ONLY the dataframe named df.
3. Use ONLY the exact column names listed above.
4. Do NOT guess column names.
5. Do NOT import anything.
6. Do NOT print anything.
7. Do NOT include explanations.
8. Do NOT include markdown.
9. Do NOT include backticks.
10. Do NOT include comments.
11. The FINAL output MUST be stored in a variable named result.

Examples of correct format:
result = df["ColumnName"].mean()
result = df.groupby("ColumnName")["OtherColumn"].sum()
result = df[["Column1", "Column2"]].corr()

Return ONLY valid Python code.
"""