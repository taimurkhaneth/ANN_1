import os
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

# 1. Load data & train model
DATA_PATH = os.path.join(os.path.dirname(__file__), "pesticides.csv")
df = pd.read_csv(DATA_PATH)

X = pd.get_dummies(df[["Area", "Year"]])
y = df["Value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
countries = sorted(df["Area"].unique())

def predict(country, year):
    data = pd.DataFrame(0, index=[0], columns=X.columns)
    data["Year"] = year
    col = "Area_" + country
    if col in data.columns:
        data[col] = 1
    result = model.predict(data)[0]
    return round(result, 2)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pesticide Use Predictor | ANN Lab Task 1</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #10b981;
            --primary-hover: #059669;
            --bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body {
            background: radial-gradient(circle at top right, #1e293b, #0f172a 80%);
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
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }
        .badge {
            display: inline-block;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
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
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 18px;
        }
        .metric-card span { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
        .metric-card strong { display: block; font-size: 1.25rem; color: #34d399; margin-top: 4px; }
        form { display: flex; flex-direction: column; gap: 18px; }
        .input-group label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 8px;
        }
        select, input {
            width: 100%;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: #fff;
            padding: 12px 14px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }
        select:focus, input:focus { border-color: var(--primary); }
        select option { background: #1e293b; color: #fff; }
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
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            text-align: center;
        }
        .result-box small { color: #34d399; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.5px; }
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
        <h1>Pesticide Use Predictor</h1>
        <p class="subtitle">Predict total pesticide use (tonnes of active ingredients) by country and year.</p>

        <div class="metrics-grid">
            <div class="metric-card">
                <span>Model R² Score</span>
                <strong>{{ "%.4f"|format(r2) }}</strong>
            </div>
            <div class="metric-card">
                <span>Mean Absolute Error</span>
                <strong>{{ "%.2f"|format(mae) }} t</strong>
            </div>
        </div>

        <form method="POST">
            <div class="input-group">
                <label for="country">Country / Area</label>
                <select name="country" id="country" required>
                    {% for c in countries %}
                    <option value="{{ c }}" {% if selected_country == c %}selected{% endif %}>{{ c }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="input-group">
                <label for="year">Target Year</label>
                <input type="number" name="year" id="year" placeholder="e.g. 2025" value="{{ selected_year or 2024 }}" required min="1950" max="2100">
            </div>
            <button type="submit">Predict Pesticide Use</button>
        </form>

        {% if result is not none %}
        <div class="result-box">
            <small>Prediction Result</small>
            <h3>{{ result }} tonnes</h3>
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
    selected_country = countries[0] if countries else ""
    selected_year = 2024

    if request.method == "POST":
        selected_country = request.form.get("country", selected_country)
        selected_year = int(request.form.get("year", 2024))
        result = predict(selected_country, selected_year)

    return render_template_string(
        HTML_TEMPLATE,
        countries=countries,
        r2=r2,
        mae=mae,
        result=result,
        selected_country=selected_country,
        selected_year=selected_year
    )

if __name__ == "__main__":
    app.run(port=5002, debug=True)
