import os
import uuid
import logging
import joblib
import pandas as pd

from io import BytesIO
from werkzeug.utils import secure_filename
from flask import (
    Flask, request, render_template, redirect,
    url_for, flash, session, jsonify
)

from supabase import create_client
from ml.insight_core import InsightEngine
from ml.forecast_engine import ForecastEngine
from chat.routes import chat_bp
from ml.red_flag_engine import generate_top_red_flags

# ===========================
# 🔧 ENV & LOGGING
# ===========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# 🚀 FLASK APP
# ===========================

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecret")

# ===========================
# 📦 SUPABASE CONFIG
# ===========================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "uploads")

supabase_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ===========================
# 🤖 LOAD ML INSIGHT ENGINE
# ===========================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "business_model.joblib"
)

ml_pipeline = joblib.load(MODEL_PATH)
insight_engine = InsightEngine(ml_pipeline)

# Forecast models (per user)
FORECAST_MODELS = {}

# ===========================
# 🏠 ROOT
# ===========================

@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("upload_page"))

# ===========================
# 🔐 LOGIN
# ===========================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        try:
            auth = supabase_client.auth.sign_in_with_password({
                "email": request.form.get("email"),
                "password": request.form.get("password")
            })

            session.clear()
            session["user_id"] = auth.user.id
            session["user_email"] = auth.user.email

            flash("✅ Login successful", "success")
            return redirect(url_for("upload_page"))

        except Exception as e:
            flash(f"❌ Login failed: {e}", "danger")

    return render_template("login.html")

# ===========================
# 📤 UPLOAD DATASET
# ===========================

from flask import request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from io import BytesIO
import pandas as pd
import uuid

@app.route("/upload", methods=["POST"])
def upload():

    if "user_id" not in session:
        return redirect(url_for("login"))

    file = request.files.get("file")
    if not file:
        flash("⚠ No file selected", "warning")
        return redirect(url_for("upload_page"))

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".csv"):
        flash("❌ Only CSV files are supported", "danger")
        return redirect(url_for("upload_page"))

    try:
        file_bytes = file.read()
        df = pd.read_csv(BytesIO(file_bytes))

        # 🔹 Save dataset to Supabase
        dataset_key = f"{session['user_id']}/{uuid.uuid4()}.csv"

        supabase_admin.storage.from_(SUPABASE_BUCKET).upload(
            dataset_key,
            file_bytes
        )

        session["dataset_key"] = dataset_key

        # 🔹 Clean Data
        df = df.dropna(how="all")
        df = df.ffill().bfill()

        # 🔹 Generate Red Flags
        red_flag_visuals = generate_top_red_flags(df)

        # 🔹 Generate Insights
        insight_result = insight_engine.run(df)

        session["dataset_summary"] = {
    "rows": df.shape[0],
    "column_count": df.shape[1],
    "columns": list(df.columns),  # 🔥 IMPORTANT
    "text_insights": insight_result["text_insights"]
    }

        session["insight_visuals"] = insight_result["visual_insights"]
        session["red_flag_visuals"] = red_flag_visuals

        flash("✅ Dataset uploaded and analyzed successfully", "success")

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"❌ Upload failed: {e}", "danger")

    return redirect(url_for("upload_page"))
# ===========================
# 📄 UPLOAD PAGE (DASHBOARD)
# ===========================

@app.route("/upload_page")
def upload_page():
    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "upload.html",
        dataset_uploads=session.get("dataset_uploads"),
        dataset_summary=session.get("dataset_summary"),
        user_email=session.get("user_email")
    )

# ===========================
# 📊 Generate graph
# ===========================

@app.route("/generate_graph", methods=["POST"])
def generate_graph():

    if "dataset_key" not in session:
        return jsonify({"error": "No dataset uploaded"}), 400

    payload = request.get_json()
    graph_type = payload.get("graph_type")
    x_axis = payload.get("x_axis")
    y_axis = payload.get("y_axis")

    file_data = supabase_admin.storage.from_(SUPABASE_BUCKET).download(
        session["dataset_key"]
    )

    df = pd.read_csv(BytesIO(file_data))

    import matplotlib.pyplot as plt

    plt.figure()

    if graph_type == "scatter":
        plt.scatter(df[x_axis], df[y_axis])
    else:
        grouped = df.groupby(x_axis)[y_axis].sum()
        grouped.plot(kind=graph_type)

    filename = f"static/insights/{uuid.uuid4()}.png"
    plt.savefig(filename)
    plt.close()

    return jsonify({"image_url": filename})
# ===========================
# 📊 INSIGHTS PAGE
# ===========================

@app.route("/insights")
def insights_page():
    if "dataset_summary" not in session:
        return redirect(url_for("upload_page"))

    return render_template(
    "insights.html",
    dataset_summary=session.get("dataset_summary"),
    insight_visuals=session.get("insight_visuals"),
    red_flag_visuals=session.get("red_flag_visuals")  # 👈 REQUIRED
)
# ===========================
# 💬 CHAT PAGE  ✅ FIXED ROUTE
# ===========================

@app.route("/chat")
def chat_page():
    if "dataset_summary" not in session:
        return redirect(url_for("upload_page"))

    return render_template("chat.html")

# ===========================
# 🔮 ESTIMATION PAGE
# ===========================

@app.route("/estimate")
def estimate_page():
    if "dataset_summary" not in session:
        return redirect(url_for("upload_page"))

    return render_template(
        "estimate.html",
        dataset_summary=session["dataset_summary"]
    )

# ===========================
# ▶ TRAIN FORECAST MODEL
# ===========================

@app.route("/train_forecast", methods=["POST"])
def train_forecast():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    payload = request.get_json() or {}
    target_col = payload.get("target")

    if not target_col:
        return jsonify({"error": "Target column is required"}), 400

    if "dataset_key" not in session:
        return jsonify({"error": "No dataset uploaded"}), 400

    try:
        file_data = supabase_admin.storage.from_(SUPABASE_BUCKET).download(
            session["dataset_key"]
        )
        df = pd.read_csv(BytesIO(file_data))

        engine = ForecastEngine.auto_train(df, target_col)
        FORECAST_MODELS[session["user_id"]] = engine

        session["forecast_info"] = {
            "target": target_col,
            "features": engine.feature_cols
        }

        return jsonify({"status": "Estimation model trained successfully"})

    except Exception as e:
        logger.exception("Training failed")
        return jsonify({"error": str(e)}), 500

# ===========================
# 🔮 RUN FORECAST
# ===========================

@app.route("/forecast", methods=["POST"])
def forecast():
    engine = FORECAST_MODELS.get(session.get("user_id"))
    if not engine:
        return jsonify({"error": "Estimation model not trained"}), 400

    payload = request.get_json() or {}

    try:
        horizon = int(payload.get("horizon", 1))
    except ValueError:
        return jsonify({"error": "Prediction window must be a number"}), 400

    try:
        result = engine.predict(horizon=horizon)
        session["forecast_result"] = result
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# 💬 CHAT API (Gemini)
# ===========================

app.register_blueprint(chat_bp, url_prefix="/chat")

# ===========================
# 🚪 LOGOUT
# ===========================

@app.route("/logout")
def logout():
    FORECAST_MODELS.pop(session.get("user_id"), None)
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for("login"))

# ===========================
# ▶ RUN
# ===========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


