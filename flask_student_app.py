import os
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

# 1. Load data & train model
DATA_PATH = os.path.join(os.path.dirname(__file__), "student-por.csv")
df = pd.read_csv(DATA_PATH)

X = df[["G1", "G2", "studytime", "failures", "absences"]]
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

def predict(g1, g2, studytime, failures, absences):
    data = pd.DataFrame([[g1, g2, studytime, failures, absences]], columns=X.columns)
    result = model.predict(data)[0]
    return round(result, 2)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Grade Predictor | ANN Lab Task 1</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg: #0b0f19;
            --card-bg: rgba(23, 31, 51, 0.75);
            --border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body {
            background: radial-gradient(circle at top right, #1e1b4b, #0b0f19 80%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }
        .container {
            width: 100%;
            max-width: 680px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 36px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }
        .badge {
            display: inline-block;
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 9999px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 16px;
        }
        h1 { font-size: 1.85rem; font-weight: 800; margin-bottom: 8px; }
        p.subtitle { color: var(--text-muted); font-size: 0.95rem; margin-bottom: 24px; }
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 24px;
        }
        .metric-card {
            background: rgba(11, 15, 25, 0.6);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 18px;
        }
        .metric-card span { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
        .metric-card strong { display: block; font-size: 1.25rem; color: #818cf8; margin-top: 4px; }
        form { display: flex; flex-direction: column; gap: 16px; }
        .input-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
        .input-group label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 8px;
        }
        input {
            width: 100%;
            background: rgba(11, 15, 25, 0.8);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: #fff;
            padding: 12px 14px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }
        input:focus { border-color: var(--primary); }
        button {
            background: linear-gradient(135deg, var(--primary), var(--primary-hover));
            color: white;
            font-weight: 700;
            border: none;
            padding: 14px;
            border-radius: 10px;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.15s, opacity 0.2s;
            margin-top: 8px;
        }
        button:hover { opacity: 0.95; transform: translateY(-1px); }
        button:active { transform: translateY(0); }
        .result-box {
            margin-top: 24px;
            padding: 20px;
            border-radius: 12px;
            background: rgba(99, 102, 241, 0.12);
            border: 1px solid rgba(99, 102, 241, 0.3);
            text-align: center;
        }
        .result-box small { color: #818cf8; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.5px; }
        .result-box h3 { font-size: 1.5rem; color: #fff; margin-top: 4px; }
        .author-footer {
            margin-top: 28px;
            text-align: center;
            font-size: 0.82rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
            padding-top: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <span class="badge">Linear Regression Model</span>
        <h1>Student Grade Predictor</h1>
        <p class="subtitle">Predict student final period exam score (G3, 0-20) from early grades & academic factors.</p>

        <div class="metrics-grid">
            <div class="metric-card">
                <span>Model R² Score</span>
                <strong>{{ "%.4f"|format(r2) }}</strong>
            </div>
            <div class="metric-card">
                <span>Mean Absolute Error</span>
                <strong>{{ "%.2f"|format(mae) }} pts</strong>
            </div>
        </div>

        <form method="POST">
            <div class="input-row">
                <div class="input-group">
                    <label for="g1">G1 Grade (0 - 20)</label>
                    <input type="number" step="any" min="0" max="20" name="g1" id="g1" value="{{ form_data.g1 if form_data else '14' }}" required>
                </div>
                <div class="input-group">
                    <label for="g2">G2 Grade (0 - 20)</label>
                    <input type="number" step="any" min="0" max="20" name="g2" id="g2" value="{{ form_data.g2 if form_data else '10' }}" required>
                </div>
            </div>

            <div class="input-row">
                <div class="input-group">
                    <label for="studytime">Study Time (1 - 4 scale)</label>
                    <input type="number" step="any" min="1" max="4" name="studytime" id="studytime" value="{{ form_data.studytime if form_data else '2' }}" required>
                </div>
                <div class="input-group">
                    <label for="failures">Past Failures (0 - 3)</label>
                    <input type="number" min="0" max="3" name="failures" id="failures" value="{{ form_data.failures if form_data else '1' }}" required>
                </div>
            </div>

            <div class="input-group">
                <label for="absences">School Absences (0 - 93)</label>
                <input type="number" min="0" max="100" name="absences" id="absences" value="{{ form_data.absences if form_data else '15' }}" required>
            </div>

            <button type="submit">Predict Final Grade (G3)</button>
        </form>

        {% if result is not none %}
        <div class="result-box">
            <small>Predicted Final Score (G3)</small>
            <h3>{{ result }} / 20</h3>
        </div>
        {% endif %}

        <div class="author-footer">
            Developed by <strong>Taimur Sultan (24108145)</strong> | SZABIST Islamabad
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = None
    if request.method == "POST":
        form_data = {
            "g1": request.form.get("g1", 14),
            "g2": request.form.get("g2", 10),
            "studytime": request.form.get("studytime", 2),
            "failures": request.form.get("failures", 1),
            "absences": request.form.get("absences", 15)
        }
        result = predict(
            float(form_data["g1"]),
            float(form_data["g2"]),
            float(form_data["studytime"]),
            float(form_data["failures"]),
            float(form_data["absences"])
        )
    return render_template_string(HTML_TEMPLATE, r2=r2, mae=mae, result=result, form_data=form_data)

if __name__ == "__main__":
    app.run(port=5003, debug=True)
