# Artificial Neural Networks (Lab) - Task # 01
### Shaheed Zulfikar Ali Bhutto Institute of Science & Technology (SZABIST), Islamabad
**Department of Robotics & Artificial Intelligence**

---

## 📌 Student & Submission Details

| Field | Information |
| :--- | :--- |
| **Student Name** | **Taimur Sultan** |
| **Registration Number** | **24108145** |
| **Degree Program / Class** | BS (AI) - 5A |
| **Department** | Department of Robotics & Artificial Intelligence |
| **Institution** | SZABIST Islamabad Campus |
| **Course** | Artificial Neural Networks (Lab) |
| **Instructor** | Mr. Hassaan Mujtaba |
| **Submission Report** | [`TASK 1(24108145)ANN .pdf`](TASK%201(24108145)ANN%20.pdf) |
| **GitHub Repository** | [https://github.com/taimurkhaneth/ANN_1](https://github.com/taimurkhaneth/ANN_1) |

---

## 🚀 Project Overview

This repository contains the complete implementation, evaluation, and interactive web deployments for **Lab Task 1** of the Artificial Neural Networks course. The objective of this lab is to explore, build, evaluate, and serve **Multiple Linear Regression** models on two distinct real-world datasets:

1. **Task 1: Agricultural Pesticide Use Prediction**
   - Predicts national pesticide consumption (`Value` in tonnes of active ingredients) based on geographical region (`Area`) and temporal trend (`Year`).
2. **Task 2: Student Final Academic Performance Prediction**
   - Predicts final period course grade (`G3`, scale 0–20) based on early period evaluations (`G1`, `G2`), weekly study habits, failure history, and school absences.

Each task is equipped with:
- Model training and validation using an **80/20 train-test split**.
- Full evaluation with **$R^2$ Score** and **Mean Absolute Error (MAE)**.
- Standalone **Gradio** interactive web interfaces.
- Standalone **Flask** web applications.
- A **Unified Master Dashboard** (`app.py`) hosting both predictive engines simultaneously.
- An end-to-end executed **Jupyter Notebook** (`ANN_Lab_Task_1_24108145.ipynb`).

---

## 📊 Summary of Model Evaluation & Results

| Task | Dataset | Input Features ($X$) | Target Variable ($y$) | $R^2$ Score | Mean Absolute Error (MAE) |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Task 1: Pesticides** | `pesticides.csv` | `Area` (One-Hot Encoded, 169 countries) + `Year` | `Value` (tonnes) | **0.9345** (93.45%) | **7,492.42 tonnes** |
| **Task 2: Student Grades** | `student-por.csv` | `G1`, `G2`, `studytime`, `failures`, `absences` | `G3` (0–20 grade) | **0.8633** (86.33%) | **0.75 grade points** |

---

## 📐 Mathematical Formulation: Multiple Linear Regression

Multiple Linear Regression models the linear relationship between a dependent target variable $y$ and a set of $n$ independent explanatory features $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n = \beta_0 + \sum_{i=1}^n \beta_i x_i$$

Where:
- $\hat{y}$ is the predicted value.
- $\beta_0$ is the bias (intercept).
- $\beta_i$ are the learned feature regression weights (coefficients).

The model parameters are optimized via Ordinary Least Squares (OLS) by minimizing the Mean Squared Error (MSE):

$$J(\boldsymbol{\beta}) = \frac{1}{2m} \sum_{k=1}^m \left( \hat{y}^{(k)} - y^{(k)} \right)^2$$

### Evaluation Metrics
- **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\sum_{k=1}^m (y^{(k)} - \hat{y}^{(k)})^2}{\sum_{k=1}^m (y^{(k)} - \bar{y})^2}$$
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{m} \sum_{k=1}^m |y^{(k)} - \hat{y}^{(k)}|$$

---

## 🌿 Task 1: Pesticide Consumption Prediction

### 1. Dataset Details
- **File:** `pesticides.csv`
- **Source:** Food and Agriculture Organization (FAO)
- **Observations:** 4,349 rows
- **Columns:** `Domain`, `Area`, `Element`, `Item`, `Year`, `Unit`, `Value`

### 2. Feature Engineering & Training
- The categorical column `Area` is encoded into dummy indicators via `pd.get_dummies()`, producing 170 input columns (`Area_*` + `Year`).
- Target variable is `Value` (pesticide quantity in tonnes).
- Train/Test Split: 80% train ($N=3,479$), 20% test ($N=870$) with `random_state=42`.

### 3. Quantitative Performance
- **$R^2$ Score:** `0.9345056989283204` ($\approx 93.45\%$)
- **Mean Absolute Error (MAE):** `7492.419186478035` tonnes
- **Inference Example:**
  - Country: `Albania`, Year: `2024` $\rightarrow$ Predicted Pesticide Consumption: `2197.80 tonnes`.

---

## 🎓 Task 2: Student Final Grade Prediction (G3)

### 1. Dataset Details
- **File:** `student-por.csv`
- **Source:** UCI Machine Learning Repository (Student Performance Dataset)
- **Observations:** 649 secondary school students
- **Columns:** 33 demographic, social, and academic indicators

### 2. Feature Selection & Training
- Features selected:
  1. `G1`: First period exam score ($0 - 20$)
  2. `G2`: Second period exam score ($0 - 20$)
  3. `studytime`: Weekly study duration ($1: <2\text{ hrs}, 2: 2-5\text{ hrs}, 3: 5-10\text{ hrs}, 4: >10\text{ hrs}$)
  4. `failures`: Number of past class failures ($0 - 3$)
  5. `absences`: Total number of school absences ($0 - 93$)
- Target variable: `G3` (Final course grade, $0 - 20$).
- Train/Test Split: 80% train ($N=519$), 20% test ($N=130$) with `random_state=42`.

### 3. Quantitative Performance & Learned Weights
- **$R^2$ Score:** `0.863263569485066` ($\approx 86.33\%$)
- **Mean Absolute Error (MAE):** `0.7462407529568897` points ($\approx 0.75$ marks)
- **Learned Regression Equation:**
  $$\hat{G3} = -0.2831 + 0.1542(G1) + 0.8988(G2) - 0.1042(\text{studytime}) - 0.1873(\text{failures}) + 0.0245(\text{absences})$$

- **Inference Example (matching report):**
  - Inputs: $G1=14$, $G2=10$, $\text{studytime}=2$, $\text{failures}=1$, $\text{absences}=15$
  - Output: $\hat{G3} = 11.20 / 20$

---

## 📁 Repository Structure

```plaintext
ANN_1/
├── app.py                         # Unified Flask Portal hosting both models (Port 5000)
├── flask_pesticides_app.py        # Dedicated Flask Web App for Pesticides Prediction (Port 5002)
├── flask_student_app.py           # Dedicated Flask Web App for Student Grade Prediction (Port 5003)
├── task1_pesticides.py            # Task 1 Python Script (Linear Regression + Gradio + Flask)
├── task2_student.py               # Task 2 Python Script (Linear Regression + Gradio + Flask)
├── ANN_Lab_Task_1_24108145.ipynb  # Executed Jupyter Notebook with both models & outputs
├── pesticides.csv                 # Pesticides dataset
├── student-por.csv                # Student performance dataset
├── TASK 1(24108145)ANN .pdf       # Official Lab Task 1 Submission Report PDF
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore patterns
└── README.md                      # Comprehensive project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/taimurkhaneth/ANN_1.git
cd ANN_1
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run the Applications

### 🌐 1. Unified Master Portal (Both Models Together)
Launch the unified web dashboard with interactive tabs:
```bash
python3 app.py
```
Open your browser at: **`http://localhost:5000`**

---

### 🌿 2. Task 1: Pesticide Use Predictor
- **Run the dedicated modern Flask app:**
  ```bash
  python3 flask_pesticides_app.py
  ```
  Visit: **`http://localhost:5002`**

- **Run via Gradio interface:**
  ```bash
  python3 task1_pesticides.py --gradio
  ```
  Visit: **`http://localhost:7860`**

---

### 🎓 3. Task 2: Student Final Grade Predictor
- **Run the dedicated modern Flask app:**
  ```bash
  python3 flask_student_app.py
  ```
  Visit: **`http://localhost:5003`**

- **Run via Gradio interface:**
  ```bash
  python3 task2_student.py --gradio
  ```
  Visit: **`http://localhost:7860`**

---

### 📓 4. Jupyter Notebook
Open and interact with the executed notebook in Jupyter Lab or Notebook:
```bash
jupyter notebook ANN_Lab_Task_1_24108145.ipynb
```

---

## 👨‍💻 Author

- **Name:** Taimur Sultan
- **Registration No:** 24108145
- **Department:** Department of Robotics and Artificial Intelligence
- **University:** SZABIST Islamabad Campus
- **GitHub:** [@taimurkhaneth](https://github.com/taimurkhaneth)
