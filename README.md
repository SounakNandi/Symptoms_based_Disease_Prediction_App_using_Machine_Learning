# 🚀 NeuralCare-Symptom

AI-powered symptom-based disease detection using Gradient-Boosting.

NeuralCare-Symptom is a **Machine Learning-based desktop application** designed to provide preliminary medical insights by predicting potential health conditions based on user symptoms, alongside recommendations for medication, nutrition, and general wellness.

> **⚠️ DISCLAIMER: This tool is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns.**

## 🎥 Demo

Check out the demo to see the app in action (version 2):

<video src="https://github.com/user-attachments/assets/f3678442-46ce-4e74-b1c0-08952da2441f" width=auto height="250" controls>
</video>

## 📌 Overview

Millions of people face difficulties accessing timely medical help due to a lack of awareness or healthcare facilities. This app aims to assist users in understanding possible health conditions based on their symptoms, offering a first level of medical insight.

This project presents an automated disease prediction system that uses Machine Learning algorithms (Gradient-Boosting) to analyze user symptoms and classify them into 41 unique disease categories. Beyond diagnosis, the system provides additional insights such as medication recommendations, nutritional advice, and wellness tips. The system was trained after testing and evaluating multiple models to ensure high-performance accuracy.

> Specifically, several candidate models—including **XGBoost, Random Forest, SVM, Decision Tree, and Neural Networks**—were trained on the same dataset and benchmarked against one another. Following this rigorous testing phase, the highest-performing model was selected for integration into the application to guarantee optimal accuracy.

## ✨ Key Features

- **Accurate Prediction**: Analyzes 130+ symptoms to identify potential health conditions.
- **Precision AI**: Powered by rigorous cross-model testing (XGBoost, Random Forest, SVM, Decision Tree, Neural Networks) to select the optimal engine.
- **Comprehensive Insights**: Delivers tailored precautions, medications, recommended diets, and workouts for each predicted condition.
- **Built-in BMI Calculator**: Automatic BMI calculation with visual health status indicators.

## �️ Tech Stack

- **Machine Learning**: Scikit-learn, XGBoost, Random Forest, SVM, Decision Tree, Neural Networks
- **Language**: Python 3.10+
- **GUI Framework**: PyQt6
- **Data Handling**: Pandas, NumPy

## 📊 Data Collection

The model is trained on a comprehensive symptom-disease dataset containing thousands of records.

- **Source**: [Kaggle - Medicine Recommendation System Dataset](https://www.kaggle.com/datasets/noorsaeed/medicine-recommendation-system-dataset)
- **Size**: 4,920+ labeled samples.
- **Symptoms**: 130+ distinct features.
- **Targets**: 41 distinct medical conditions.

## 🔧 Installation & Usage

1️⃣ Clone the Repository

```bash
git clone https://github.com/SounakNandi/NeuralCare-Symptom.git
cd NeuralCare-Symptom
```

2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

3️⃣ Run the Application

**Version 2 (recommended)**

```bash
python ui/app2.py
```

**Version 1**

```bash
python ui/app1.py
```

## Credits

Made with ❤️ by some cool guy [SOUNAK NANDI](https://github.com/SounakNandi)
