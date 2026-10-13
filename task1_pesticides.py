# Dataset 1 Pesticides Data -> Predict pesticide use "Value" (tonnes)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load the data
df = pd.read_csv("pesticides.csv")
print(df.head())

# 2. Choose input columns (X) and target column (y)
# "Area" (country) is text, so get_dummies turns it into 0/1 columns
X = pd.get_dummies(df[["Area", "Year"]])
y = df["Value"]

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
countries = sorted(df["Area"].unique())

def predict(country, year):
    data = pd.DataFrame(0, index=[0], columns=X.columns) # one row of zeros
    data["Year"] = year
    data["Area_" + country] = 1 # mark the chosen country
    result = model.predict(data)[0]
    return round(result, 2)

# ===== 7. Gradio Interface =====
import gradio as gr

app_gradio = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(countries, label="Country"),
        gr.Number(label="Year"),
    ],
    outputs=gr.Number(label="Predicted Pesticide Use (tonnes)"),
    title="Pesticide Use Predictor (Linear Regression)",
)

# ===== 8. Flask Interface =====
from flask import Flask, request, render_template_string

app = Flask(__name__)

html = """
<h2>Pesticide Use Predictor (Linear Regression)</h2>
<form method="post">
  Country:
  <select name="country">
    {% for c in countries %}
    <option>{{ c }}</option>
    {% endfor %}
  </select><br><br>
  Year: <input name="year" required><br><br>
  <button type="submit">Predict</button>
</form>
<h3>{{ result }}</h3>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        country = request.form["country"]
        year = int(request.form["year"])
        result = "Predicted Pesticide Use: " + str(predict(country, year)) + " tonnes"
    return render_template_string(html, result=result, countries=countries)

if __name__ == "__main__":
    import sys
    if "--gradio" in sys.argv:
        app_gradio.launch()
    else:
        print("Starting Flask app on port 5002... (Run with --gradio to launch Gradio instead)")
        app.run(port=5002)
