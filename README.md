# Breast Cancer Prediction Using Machine Learning

## Project Overview

This project is a machine learning-powered web application that predicts whether a breast tumour is **Malignant (Cancerous)** or **Benign (Non-cancerous)** based on five selected tumour measurements.

The application uses a **Random Forest Classifier** implemented with Scikit-learn and integrated into a Flask web application. Users can enter tumour measurements through a responsive web interface and receive a prediction alongside a model confidence estimate.

The application is deployed on **Render**, making it accessible online without requiring users to run the application locally.

## Live Demo

**Try the application:** [Insert Your Render Deployment URL Here]

## Features

- **Data Processing:** Loads and preprocesses the breast cancer dataset using Pandas.
- **Feature Selection:** Uses five numerical tumour measurements.
- **Model Training:** Implements a Random Forest Classifier with 500 estimators.
- **Model Persistence:** Saves and loads the trained model using Joblib to avoid retraining on every prediction request.
- **Prediction:** Classifies tumour measurements as malignant or benign.
- **Probability Estimation:** Uses `predict_proba()` to estimate the model's probability for its predicted class.
- **Input Validation:** Validates numerical inputs and handles invalid or missing values.
- **Web Application:** Provides an interactive interface built with Flask, HTML5, and CSS3.
- **Cloud Deployment:** Hosted on Render and accessible through a public deployment URL.

## Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computing |
| Scikit-learn | Machine learning |
| Random Forest | Classification algorithm |
| Joblib | Model serialization and persistence |
| Flask | Backend web framework |
| HTML5 | Frontend structure |
| CSS3 | Frontend styling and responsiveness |
| Gunicorn | Production WSGI server |
| Render | Cloud hosting and deployment |
| Git & GitHub | Version control and source code management |

## Dataset

The project uses a breast cancer dataset hosted on GitHub.

**Dataset:** [Breast Cancer Dataset](https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/master/breast-cancer-data.csv)

The model uses five selected features:

1. `radius_mean`
2. `texture_mean`
3. `perimeter_mean`
4. `smoothness_mean`
5. `compactness_mean`

**Target variable:**

- `M` = Malignant (1)
- `B` = Benign (0)

## Machine Learning Workflow

The machine learning pipeline follows these steps:

1. Load the dataset using Pandas.
2. Select the five relevant input features and diagnosis target.
3. Encode diagnosis labels into numerical values.
4. Remove rows with missing values.
5. Initialize a Random Forest Classifier.
6. Train the model using the selected features.
7. Save the trained model using Joblib.
8. Load the saved model when the application starts.
9. Accept new measurements through the Flask web interface.
10. Validate user inputs and generate predictions.
11. Calculate the predicted class probability using `predict_proba()`.

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=1,
    class_weight="balanced"
)
```

### Model Persistence

The trained model is saved using Joblib:

```python
joblib.dump(model, MODEL_PATH)
```

The application checks whether a saved model exists before deciding to train a new one.

## Project Structure

```text
Breast-Cancer-Prediction/
│
├── app.py
├── train.py
├── requirements.txt
│
├── templates/
│   └── home.html
│
└── README.md
```

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/Breast-Cancer-Prediction.git
```

### 2. Navigate into the project directory

```bash
cd Breast-Cancer-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000/
```

## Render Cloud Deployment

The Flask application is deployed on **Render**, a cloud platform that supports hosting Python web applications.

### Deployment Configuration

| Setting | Configuration |
|---|---|
| Platform | Render |
| Service Type | Web Service |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app` |
| Source Code | GitHub |

### Deployment Workflow

1. Pushed the application source code to GitHub.
2. Connected the GitHub repository to Render.
3. Created a Render Web Service.
4. Configured the Python environment and build command.
5. Configured Gunicorn as the production application server.
6. Deployed the application.
7. Tested the live application through its public URL.

## Future Improvements

- Add confusion matrix, precision, recall, and F1-score evaluation.
- Perform hyperparameter tuning to improve model performance.
- Compare Random Forest with Logistic Regression, SVM, and other classification algorithms.
- Implement cross-validation for more robust model evaluation.
- Add interactive visualizations for prediction results.
- Improve model monitoring and deployment reliability.
- Configure a custom domain and HTTPS support.

## Disclaimer

This application is developed strictly for educational and machine learning demonstration purposes.

Its predictions are not medical diagnoses and should not be used to make clinical decisions. The model's confidence estimate does not represent medical certainty. Always consult qualified healthcare professionals for medical evaluation.

## Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning Enthusiast | Python Developer

- GitHub: [@Vedline2547](https://github.com/Vedline2547)

---

*Building practical machine learning applications through Python, web development, and cloud computing.*
