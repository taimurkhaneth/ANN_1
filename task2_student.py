# Dataset 2 Student Performance -> Predict final grade "G3"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load the data
df = pd.read_csv("student-por.csv")
print(df.head())

# 2. Choose input columns (X) and target column (y)
X = df[["G1", "G2", "studytime", "failures", "absences"]]
y = df["G3"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Check how good the model is
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# 6. Prediction function (used by both Gradio and Flask)
def predict(g1, g2, studytime, failures, absences):
    data = pd.DataFrame([[g1, g2, studytime, failures, absences]], columns=X.columns)
    result = model.predict(data)[0]
    return round(result, 2)

# ===== 7. Gradio Interface =====
import gradio as gr

app_gradio = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="G1 (First period grade, 0-20)"),
        gr.Number(label="G2 (Second period grade, 0-20)"),
        gr.Number(label="Study Time (1-4)"),
        gr.Number(label="Past Failures (0-3)"),
        gr.Number(label="Absences"),
    ],
    outputs=gr.Number(label="Predicted Final Grade (G3)"),
    title="Student Final Grade Predictor (Linear Regression)",
)

# ===== 8. Flask Interface =====
from flask import Flask, request, render_template_string

app = Flask(__name__)

html = """
<h2>Student Final Grade Predictor (Linear Regression)</h2>
<form method="post">
  G1 (First period grade, 0-20): <input name="g1" required><br><br>
  G2 (Second period grade, 0-20): <input name="g2" required><br><br>
  Study Time (1-4): <input name="studytime" required><br><br>
  Past Failures (0-3): <input name="failures" required><br><br>
  Absences: <input name="absences" required><br><br>
  <button type="submit">Predict</button>
</form>
<h3>{{ result }}</h3>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        g1 = float(request.form["g1"])
        g2 = float(request.form["g2"])
        studytime = float(request.form["studytime"])
        failures = float(request.form["failures"])
        absences = float(request.form["absences"])
        result = "Predicted Final Grade (G3): " + str(predict(g1, g2, studytime, failures, absences))
    return render_template_string(html, result=result)

if __name__ == "__main__":
    import sys
    if "--gradio" in sys.argv:
        app_gradio.launch()
    else:
        print("Starting Flask app on port 5003... (Run with --gradio to launch Gradio instead)")
        app.run(port=5003)
