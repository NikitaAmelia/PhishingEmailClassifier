# Phishing Email Classifier

A **Machine Learning–based phishing detection system** that classifies emails as **phishing** or **legitimate** based on their text content. This project applies natural language processing (NLP) techniques and is deployed as an interactive **Streamlit web application**.

---

## 🔗 Live Demo (Streamlit App)

👉 [https://phishingemailclassifier.streamlit.app/]

---

## 📌 Project Overview

The goal of this project is to build an automated phishing email detection system using **TF-IDF** for text vectorization and a **Random Forest Classifier** for prediction. The system helps identify potentially malicious emails based on their content.

Key features:

* Text preprocessing and feature extraction using TF-IDF
* Supervised machine learning with Random Forest
* Saved and reusable trained model
* Interactive Streamlit-based web interface

---

## 📂 Dataset

The dataset used in this project is sourced from **Kaggle**.

🔗 Dataset link:
[https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=phishing_email.csv](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=phishing_email.csv)

### How to Download the Dataset

1. Open the dataset link on Kaggle
2. Log in with your Kaggle account
3. Click **Download**
4. Extract and place the dataset file inside the `/dataset` folder

---

## 🛠 Libraries & Environment

### Recommended Python Version

* **Python 3.10+**

### Install Dependencies

All required libraries are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
streamlit
scikit-learn
pandas
numpy
joblib
```

---

## 📁 Project Structure

```
phishing_email_classifier/
│
├── models/
│   └── random_forest_tfidf_model.pkl     # trained & saved model
│
├── src/
│   ├── preprocessing.py                  # text preprocessing functions
│   ├── predict.py                        # model loading & prediction logic
│
├── app/
│   └── app.py                            # Streamlit application
│
├── requirements.txt                      # project dependencies
├── README.md                             # project documentation
```

---

## 🔍 Model Training & Evaluation

If you want to retrain the model or experiment with the notebook:

1. Ensure the dataset is placed in the `/dataset` directory
2. Activate your Python environment (optional):

   ```bash
   conda activate env_name
   ```

   or

   ```bash
   source venv/bin/activate
   ```
3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```
4. Open the notebook:

   ```
   Phishing_Email_Classifier.ipynb
   ```
5. Run all cells to train and evaluate the model
6. Save the trained model to:

   ```
   models/random_forest_tfidf_model.pkl
   ```

---

## 🚀 Run the Streamlit App Locally

From the project root directory, run:

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

If successful, a local URL will be generated and opened in your browser.

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. It should not be used as the sole method for detecting phishing emails in real-world security-critical systems.

---

## ⭐ Future Improvements

* Experiment with advanced NLP models (e.g., BERT)
* Add email metadata features
* Improve model evaluation and reporting
* Deploy using Docker or cloud services

---

## 👩‍💻 Author

Nikita Amelia Valencia  
Machine Learning & Data Science Enthusiast

---

## 📜 License

This project is for educational and portfolio purposes.
