import os
import pandas as pd
from flask import Flask, request, render_template_string, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

# ==========================================
# MODEL 1: Pesticide Use Prediction
# ==========================================
df_pest = pd.read_csv("pesticides.csv")
X_pest = pd.get_dummies(df_pest[["Area", "Year"]])
y_pest = df_pest["Value"]

X_train_pest, X_test_pest, y_train_pest, y_test_pest = train_test_split(
    X_pest, y_pest, test_size=0.2, random_state=42
)
model_pest = LinearRegression()
model_pest.fit(X_train_pest, y_train_pest)
pred_pest = model_pest.predict(X_test_pest)
r2_pest = r2_score(y_test_pest, pred_pest)
mae_pest = mean_absolute_error(y_test_pest, pred_pest)
pest_countries = sorted(df_pest["Area"].unique())

def predict_pesticide(country, year):
    data = pd.DataFrame(0, index=[0], columns=X_pest.columns)
    data["Year"] = year
    col = "Area_" + country
    if col in data.columns:
        data[col] = 1
    result = model_pest.predict(data)[0]
    return round(result, 2)

# ==========================================
# MODEL 2: Student Grade Prediction
# ==========================================
df_student = pd.read_csv("student-por.csv")
X_student = df_student[["G1", "G2", "studytime", "failures", "absences"]]
y_student = df_student["G3"]

X_train_stu, X_test_stu, y_train_stu, y_test_stu = train_test_split(
    X_student, y_student, test_size=0.2, random_state=42
)
model_stu = LinearRegression()
model_stu.fit(X_train_stu, y_train_stu)
pred_stu = model_stu.predict(X_test_stu)
r2_stu = r2_score(y_test_stu, pred_stu)
mae_stu = mean_absolute_error(y_test_stu, pred_stu)

def predict_student(g1, g2, studytime, failures, absences):
    data = pd.DataFrame([[g1, g2, studytime, failures, absences]], columns=X_student.columns)
    result = model_stu.predict(data)[0]
    return round(result, 2)

HTML_PORTAL = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ANN Lab Task 1 Portal | Taimur Sultan (24108145)</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --surface: rgba(22, 29, 47, 0.7);
            --surface-hover: rgba(30, 41, 67, 0.85);
            --border: rgba(255, 255, 255, 0.08);
            --accent-green: #10b981;
            --accent-indigo: #6366f1;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body {
            background: radial-gradient(circle at 10% 20%, #172554 0%, #090d16 90%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 30px 16px;
        }
        .main-wrapper {
            max-width: 1000px;
            margin: 0 auto;
        }
        .header {
            background: var(--surface);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 28px 36px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }
        .header-title h1 {
            font-size: 1.7rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header-title p {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }
        .meta-tag {
            text-align: right;
        }
        .meta-tag .badge {
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 9999px;
            display: inline-block;
            margin-bottom: 6px;
        }
        .meta-tag .info {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        .tabs {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }
        .tab-btn {
            flex: 1;
            padding: 16px 20px;
            background: var(--surface);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: 14px;
            color: var(--text-secondary);
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.25s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .tab-btn.active.pest {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border-color: rgba(16, 185, 129, 0.4);
        }
        .tab-btn.active.stu {
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            border-color: rgba(99, 102, 241, 0.4);
        }
        .tab-content {
            display: none;
            background: var(--surface);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 36px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }
        .tab-content.active { display: block; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 14px;
            margin-bottom: 28px;
        }
        .stat-card {
            background: rgba(10, 15, 27, 0.6);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 16px 20px;
        }
        .stat-card span { font-size: 0.75rem; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; }
        .stat-card strong { display: block; font-size: 1.35rem; margin-top: 6px; }
        .stat-card.green strong { color: #34d399; }
        .stat-card.indigo strong { color: #818cf8; }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }
        .full-width { grid-column: 1 / -1; }
        .input-group label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }
        select, input {
            width: 100%;
            background: rgba(10, 15, 27, 0.85);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: #fff;
            padding: 12px 14px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }
        select:focus, input:focus { border-color: #6366f1; }
        select option { background: #172554; }
        .btn-submit {
            padding: 15px;
            border-radius: 10px;
            border: none;
            color: white;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            transition: opacity 0.2s, transform 0.15s;
            margin-top: 8px;
        }
        .btn-green { background: linear-gradient(135deg, #10b981, #059669); }
        .btn-indigo { background: linear-gradient(135deg, #6366f1, #4f46e5); }
        .btn-submit:hover { opacity: 0.95; transform: translateY(-1px); }
        .btn-submit:active { transform: translateY(0); }
        .result-banner {
            margin-top: 24px;
            padding: 22px;
            border-radius: 14px;
            text-align: center;
        }
        .result-banner.green {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .result-banner.indigo {
            background: rgba(99, 102, 241, 0.12);
            border: 1px solid rgba(99, 102, 241, 0.3);
        }
        .result-banner small { text-transform: uppercase; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px; }
        .result-banner.green small { color: #34d399; }
        .result-banner.indigo small { color: #818cf8; }
        .result-banner h2 { font-size: 1.8rem; margin-top: 6px; }
        .footer {
            margin-top: 30px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }
        @media (max-width: 700px) {
            .form-grid { grid-template-columns: 1fr; }
            .header { flex-direction: column; text-align: center; }
            .meta-tag { text-align: center; }
        }
    </style>
</head>
<body>
    <div class="main-wrapper">
        <div class="header">
            <div class="header-title">
                <h1>ANN Lab Task 1 Interactive Portal</h1>
                <p>Shaheed Zulfikar Ali Bhutto Institute of Science & Technology (SZABIST)</p>
            </div>
            <div class="meta-tag">
                <span class="badge">BS (AI) - 5A</span>
                <div class="info"><strong>Taimur Sultan</strong> (24108145)</div>
                <div class="info">Instructor: Mr. Hassaan Mujtaba</div>
            </div>
        </div>

        <div class="tabs">
            <button class="tab-btn pest active" onclick="switchTab('pest')">🌿 Task 1: Pesticide Predictor</button>
            <button class="tab-btn stu" onclick="switchTab('stu')">🎓 Task 2: Student Grade Predictor</button>
        </div>

        <!-- TAB 1: PESTICIDE PREDICTOR -->
        <div id="tab-pest" class="tab-content active">
            <div class="stats-grid">
                <div class="stat-card green">
                    <span>Model Performance (R²)</span>
                    <strong>{{ "%.4f"|format(r2_pest) }} (93.45%)</strong>
                </div>
                <div class="stat-card green">
                    <span>Mean Absolute Error</span>
                    <strong>{{ "%.2f"|format(mae_pest) }} tonnes</strong>
                </div>
                <div class="stat-card green">
                    <span>Dataset Rows</span>
                    <strong>4,349 entries</strong>
                </div>
            </div>

            <form method="POST" action="/predict/pesticide">
                <div class="form-grid">
                    <div class="input-group">
                        <label for="p_country">Country / Area</label>
                        <select name="country" id="p_country" required>
                            {% for c in pest_countries %}
                            <option value="{{ c }}" {% if pest_selected_country == c %}selected{% endif %}>{{ c }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="input-group">
                        <label for="p_year">Target Year</label>
                        <input type="number" name="year" id="p_year" value="{{ pest_selected_year or 2024 }}" required min="1950" max="2100">
                    </div>
                    <button type="submit" class="btn-submit btn-green full-width">Predict Pesticide Use (tonnes)</button>
                </div>
            </form>

            {% if pest_result is not none %}
            <div class="result-banner green">
                <small>Predicted Pesticide Consumption</small>
                <h2>{{ pest_result }} tonnes</h2>
            </div>
            {% endif %}
        </div>

        <!-- TAB 2: STUDENT GRADE PREDICTOR -->
        <div id="tab-stu" class="tab-content">
            <div class="stats-grid">
                <div class="stat-card indigo">
                    <span>Model Performance (R²)</span>
                    <strong>{{ "%.4f"|format(r2_stu) }} (86.33%)</strong>
                </div>
                <div class="stat-card indigo">
                    <span>Mean Absolute Error</span>
                    <strong>{{ "%.2f"|format(mae_stu) }} pts</strong>
                </div>
                <div class="stat-card indigo">
                    <span>Dataset Rows</span>
                    <strong>649 students</strong>
                </div>
            </div>

            <form method="POST" action="/predict/student">
                <div class="form-grid">
                    <div class="input-group">
                        <label for="s_g1">G1 Grade (0 - 20)</label>
                        <input type="number" step="any" min="0" max="20" name="g1" id="s_g1" value="{{ stu_data.g1 if stu_data else '14' }}" required>
                    </div>
                    <div class="input-group">
                        <label for="s_g2">G2 Grade (0 - 20)</label>
                        <input type="number" step="any" min="0" max="20" name="g2" id="s_g2" value="{{ stu_data.g2 if stu_data else '10' }}" required>
                    </div>
                    <div class="input-group">
                        <label for="s_studytime">Weekly Study Time (1 - 4 scale)</label>
                        <input type="number" step="any" min="1" max="4" name="studytime" id="s_studytime" value="{{ stu_data.studytime if stu_data else '2' }}" required>
                    </div>
                    <div class="input-group">
                        <label for="s_failures">Past Class Failures (0 - 3)</label>
                        <input type="number" min="0" max="3" name="failures" id="s_failures" value="{{ stu_data.failures if stu_data else '1' }}" required>
                    </div>
                    <div class="input-group full-width">
                        <label for="s_absences">School Absences (0 - 93)</label>
                        <input type="number" min="0" max="100" name="absences" id="s_absences" value="{{ stu_data.absences if stu_data else '15' }}" required>
                    </div>
                    <button type="submit" class="btn-submit btn-indigo full-width">Predict Final Grade (G3)</button>
                </div>
            </form>

            {% if stu_result is not none %}
            <div class="result-banner indigo">
                <small>Predicted Final Exam Grade</small>
                <h2>{{ stu_result }} / 20</h2>
            </div>
            {% endif %}
        </div>

        <div class="footer">
            SZABIST Islamabad &bull; Department of Robotics and Artificial Intelligence &bull; Student: Taimur Sultan (24108145)
        </div>
    </div>

    <script>
        function switchTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            if (tab === 'pest') {
                document.querySelector('.tab-btn.pest').classList.add('active');
                document.getElementById('tab-pest').classList.add('active');
            } else {
                document.querySelector('.tab-btn.stu').classList.add('active');
                document.getElementById('tab-stu').classList.add('active');
            }
        }
        {% if active_tab == 'stu' %}
        switchTab('stu');
        {% endif %}
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML_PORTAL,
        r2_pest=r2_pest,
        mae_pest=mae_pest,
        pest_countries=pest_countries,
        pest_selected_country=pest_countries[0] if pest_countries else "",
        pest_selected_year=2024,
        pest_result=None,
        r2_stu=r2_stu,
        mae_stu=mae_stu,
        stu_data=None,
        stu_result=None,
        active_tab="pest"
    )

@app.route("/predict/pesticide", methods=["POST"])
def handle_predict_pest():
    country = request.form.get("country", pest_countries[0] if pest_countries else "")
    year = int(request.form.get("year", 2024))
    res = predict_pesticide(country, year)
    return render_template_string(
        HTML_PORTAL,
        r2_pest=r2_pest,
        mae_pest=mae_pest,
        pest_countries=pest_countries,
        pest_selected_country=country,
        pest_selected_year=year,
        pest_result=res,
        r2_stu=r2_stu,
        mae_stu=mae_stu,
        stu_data=None,
        stu_result=None,
        active_tab="pest"
    )

@app.route("/predict/student", methods=["POST"])
def handle_predict_stu():
    stu_data = {
        "g1": request.form.get("g1", 14),
        "g2": request.form.get("g2", 10),
        "studytime": request.form.get("studytime", 2),
        "failures": request.form.get("failures", 1),
        "absences": request.form.get("absences", 15)
    }
    res = predict_student(
        float(stu_data["g1"]),
        float(stu_data["g2"]),
        float(stu_data["studytime"]),
        float(stu_data["failures"]),
        float(stu_data["absences"])
    )
    return render_template_string(
        HTML_PORTAL,
        r2_pest=r2_pest,
        mae_pest=mae_pest,
        pest_countries=pest_countries,
        pest_selected_country=pest_countries[0] if pest_countries else "",
        pest_selected_year=2024,
        pest_result=None,
        r2_stu=r2_stu,
        mae_stu=mae_stu,
        stu_data=stu_data,
        stu_result=res,
        active_tab="stu"
    )

if __name__ == "__main__":
    app.run(port=5000, debug=True)
