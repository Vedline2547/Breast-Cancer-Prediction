# Breast Cancer Prediction Using Machine Learning

## Project Overview

This project is a machine learning-powered web application that predicts whether a breast tumour is **Malignant (Cancerous)** or **Benign (Non-cancerous)** based on five tumour measurements.

The application uses a **Random Forest Classifier** implemented with Scikit-learn and integrated into a Flask web application. Users can enter tumour measurements through a responsive web interface and receive a prediction with a probability estimate for the malignant class.

The application was deployed on an **AWS EC2 instance**, demonstrating practical experience in machine learning, Python web development, Linux server management, and cloud deployment.

## Features

- **Data Processing:** Loads and preprocesses the breast cancer dataset using Pandas.
- **Feature Selection:** Uses five numerical tumour measurements.
- **Model Training:** Implements a Random Forest Classifier with 500 estimators.
- **Model Evaluation:** Uses an 80/20 train-test split and calculates classification accuracy.
- **Prediction:** Classifies tumour measurements as malignant or benign.
- **Probability Estimation:** Uses `predict_proba()` to estimate the probability of the malignant class.
- **Web Application:** Provides an interactive interface built with Flask, HTML5, and CSS3.
- **Cloud Deployment:** Hosted on AWS EC2 and accessed through the instance's public IPv4 address.

## Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computing |
| Matplotlib | Data visualization library |
| Scikit-learn | Machine learning and model evaluation |
| Random Forest | Classification algorithm |
| Flask | Backend web framework |
| HTML5 | Frontend structure |
| CSS3 | Frontend styling |
| AWS EC2 | Cloud hosting |
| Linux | Remote server environment |
| PuTTY | SSH connection and server management |
| WinSCP | Secure file transfer |

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
2. Encode diagnosis labels into numerical values.
3. Select five relevant input features.
4. Separate independent variables (X) and target variable (y).
5. Split the dataset into training (80%) and testing (20%) sets.
6. Train a Random Forest Classifier with 500 estimators.
7. Generate predictions using the test dataset.
8. Evaluate model accuracy using `accuracy_score`.
9. Generate predictions and probability estimates for new measurements.

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=500,
    n_jobs=-1
)
```

Train-test split:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

## Project Structure

```text
Breast-Cancer-Prediction/
│
├── app.py
│
├── train.py
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
pip install flask pandas numpy matplotlib scikit-learn
```

### 5. Run the application

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000/
```

## AWS EC2 Cloud Deployment

The Flask application was deployed on an **Amazon Web Services (AWS) EC2 instance**, allowing it to run on a remote Linux server.

### Deployment Tools

- **AWS EC2:** Provided the cloud computing infrastructure for hosting the application.
- **PuTTY:** Used to establish SSH connections and manage the remote Linux server.
- **WinSCP:** Used to transfer project files from the local Windows computer to the EC2 instance.
- **Linux:** Provided the server environment for running the Flask application.

### Deployment Workflow

1. Created and configured an AWS EC2 instance.
2. Connected to the remote Linux server using PuTTY through SSH.
3. Used WinSCP to transfer the project files to the EC2 instance.
4. Set up a Python virtual environment on the server.
5. Installed the required Python dependencies.
6. Executed the Flask application on the EC2 instance.
7. Accessed the deployed application externally through the EC2 instance's public IPv4 address.

### Accessing the Deployed Application

The application was accessed through the EC2 instance's public IP address and Flask's configured port.

```text
http://<EC2-PUBLIC-IP>:5000/
```

**Note:** The EC2 security group must allow inbound traffic on port 5000 for external access. For production deployment, a production WSGI server and HTTPS configuration are recommended.

## Future Improvements

- Save and load the trained model using Joblib instead of retraining on every prediction request.
- Add confusion matrix, precision, recall, and F1-score evaluation.
- Perform hyperparameter tuning to improve model performance.
- Compare Random Forest with Logistic Regression, SVM, and other classification algorithms.
- Implement input validation and error handling.
- Deploy using Gunicorn and Nginx.
- Configure a custom domain and HTTPS support.
- Improve prediction visualization and user experience.

## Disclaimer

This application is developed strictly for educational and machine learning demonstration purposes.

Its predictions are not medical diagnoses and should not be used to make clinical decisions. Always consult qualified healthcare professionals for medical evaluation.

## Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning Enthusiast | Python Developer

- GitHub: [@Vedline2547](https://github.com/Vedline2547)

---

*Building practical machine learning applications through Python, web development, and cloud computing.*
