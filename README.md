# Placement Prediction Using Machine Learning

This project leverages machine learning to predict student placement outcomes and expected salaries during campus recruitment. By analyzing various student features, the project aims to assist students and educational institutions in improving placement strategies and preparing for recruitment processes.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [Contributing](#contributing)
- [License](#license)

---

## 🌟 Overview
Campus placement is a critical milestone for students and educational institutions. This project uses Random Forest classifiers to predict:
1. **Placement Probability**: Whether a student is likely to be placed.
2. **Expected Salary**: The potential salary for placed students.

By analyzing academic performance, skills, and other relevant features, the project provides actionable insights to improve placement outcomes.

---

## 📊 Dataset
The dataset includes the following features:
- **Academic Performance**: CGPA, percentage scores, etc.
- **Skills**: Number and type of skills.
- **Extracurricular Activities**: Participation in internships, hackathons, workshops, etc.
- **Other Features**: Backlogs, mini-projects, communication skills, etc.

These features are used to train models for predicting placement status and salary.

---

## ⚙️ Installation
Follow these steps to set up and run the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Simrannaroraa/placement_prediction.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

---

## 🗂️ Project Structure
- `app.py`: Main Streamlit application.
- `model.pkl`: Placement prediction model.
- `model1.pkl`: Salary prediction model.
- `Placement_prediction_data.csv`: Dataset for placement prediction.
- `Salary_prediction_data.csv`: Dataset for salary prediction.
- `Placement_prediction.py`: Script for training the placement prediction model.
- `Salary_prediction.py`: Script for training the salary prediction model.
- `preprocessing.ipynb`: Jupyter Notebook for data preprocessing.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.

---

## 🛠️ Data Preprocessing
The preprocessing pipeline includes:
1. Handling missing values.
2. Encoding categorical variables.
3. Scaling numerical features.
4. Selecting relevant features for training.

---

## 🤖 Model Training
Two Random Forest classifiers were trained:
1. **Placement Prediction Model**: Predicts whether a student will be placed.
2. **Salary Prediction Model**: Predicts the expected salary for placed students.

### Training Steps:
- Splitting the dataset into training and testing sets.
- Initializing and training the Random Forest classifiers.
- Fine-tuning hyperparameters using Grid Search or Random Search.

---

## 📈 Evaluation
The models were evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC AUC Score**

---

## 🏆 Results
### Placement Prediction Model:
- **Accuracy**: 88.7%
- **Precision**: 0.93
- **Recall**: 0.86
- **F1 Score**: 0.90
- **ROC AUC Score**: 0.94

---

## 🚀 Streamlit App
The trained models are deployed using a Streamlit application. The app provides an interactive interface for users to:
1. Input student details.
2. Predict placement probability.
3. Estimate expected salary.

### Run the App:
```bash
streamlit run app.py
```

---

## 🤝 Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

---

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to reach out with any questions or suggestions!
