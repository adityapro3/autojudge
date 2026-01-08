# ⚖️ AutoJudge: Codeforces Problem Difficulty Predictor

## 📌 Project Overview

AutoJudge is an AI-powered web application that predicts the difficulty level and rating score of Codeforces programming problems. The system uses Natural Language Processing (NLP) and Machine Learning to analyze problem statements and automatically classify them into difficulty categories (Easy, Medium, Hard) while also predicting numerical ratings.

**Key Features:**
- Predicts problem difficulty class (Easy/Medium/Hard)
- Predicts numerical rating score (e.g., 800, 1200, 1500)
- Interactive web interface built with Streamlit
- Real-time predictions with confidence scores
- Prediction history tracking

---

## 📊 Dataset Used

**Source:** Hugging Face Dataset - `open-r1/codeforces`

**Dataset Statistics:**
- **Training Set:** 7736 problems (80% split)
- **Test Set:** 1,820 problems (20% split)
- **Additional Test Set:** 444 problems

**Features:**
- `title` - Problem title
- `description` - Problem statement
- `inputformat` - Input format description
- `outputformat` - Output format description
- `rating` - Codeforces difficulty rating
- `problemclass` - Difficulty classification (Easy/Medium/Hard)

**Preprocessing:**
1. Combined all text fields (title, description, input/output formats)
2. Text cleaning: lowercase conversion, punctuation removal, whitespace normalization
3. TF-IDF vectorization with 5000 max features and English stop words removal

---

## 🤖 Approach and Models Used

### Text Preprocessing Pipeline
```
Raw Text → Lowercase → Remove Punctuation → Remove Extra Spaces → TF-IDF Vectorization
```

### Model Architecture

#### 1. Classification Model (Difficulty Prediction)
- **Algorithm:** Random Forest Classifier
- **Number of Estimators:** 1000 trees
- **Features:** TF-IDF vectors (max 5000 features)
- **Classes:** Easy, Medium, Hard
- **Purpose:** Classify problems into difficulty categories

#### 2. Regression Model (Rating Prediction)
- **Algorithm:** Random Forest Regressor
- **Number of Estimators:** 1000 trees
- **Features:** TF-IDF vectors (max 5000 features)
- **Output:** Numerical rating score
- **Purpose:** Predict exact difficulty rating

### Feature Engineering
- **TF-IDF Vectorization:** Converts text to numerical features
- **Max Features:** 5000 most important words
- **Stop Words:** English stop words removed
- **Combined Text:** All problem fields merged for comprehensive analysis

---

## 📈 Evaluation Metrics

### Classification Model Performance

**Overall Accuracy:** 60% (Test Set 1) | 64% (Test Set 2)

**Detailed Classification Report (Test Set 2):**
```
              precision    recall    f1-score    support
        Easy      0.74      0.59      0.66        116
      Medium      0.37      0.44      0.40        112
        Hard      0.75      0.76      0.76        216

    accuracy                          0.64        444
   macro avg      0.62      0.60      0.61        444
weighted avg      0.65      0.64      0.64        444
```

**Key Insights:**
- Best performance on Hard problems (F1: 0.76)
- Moderate performance on Easy problems (F1: 0.66)
- Lowest performance on Medium problems (F1: 0.40)

### Regression Model Performance

**Mean Absolute Error (MAE):**
- Test Set 1: **470.80** points
- Test Set 2: **508.78** points

**Interpretation:** On average, the predicted rating is off by approximately 470-508 points from the actual rating.

---

## 🚀 Steps to Run the Project Locally

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

**1. Clone the repository**
```bash
git clone https://github.com/adityapro3/autojudge.git
cd autojudge
```

**2. Install required dependencies**
```bash
pip install -r requirements.txt
```

**3. Ensure model files are present**
Make sure these files exist in the project directory:
- `model_class.pkl` (Classification model)
- `model_score.pkl` (Regression model)

**4. Train models (if not available)**
```bash
jupyter notebook Auto_Edge_Notebook.ipynb
# Run all cells to train and save models
```

**5. Run the Streamlit application**
```bash
streamlit run app.py
```

**6. Open in browser**
```
The app will automatically open at: http://localhost:8501
```

---

## 🖥️ Web Interface Explanation

### Main Interface Components

#### 1. **Sidebar**
- **Quick Stats:** Displays total predictions made and most common difficulty class
- **Clear History Button:** Resets prediction history

#### 2. **Input Form**
- **Problem Title (Required):** Enter the problem name
- **Problem Description (Required):** Paste the main problem statement
- **Input Description (Optional):** Describe input format and constraints
- **Output Description (Optional):** Describe expected output format
- **Character Counter:** Shows total characters entered

#### 3. **Prediction Results**
- **Difficulty Class:** Visual indicator with color coding
  - 🟢 Green: Easy
  - 🟡 Yellow: Medium
  - 🔴 Red: Hard
- **Rating Score:** Predicted numerical rating (rounded to nearest 100)
- **Progress Bar:** Visual representation of difficulty level
- **Confidence Scores:** Model confidence for each class (when available)

#### 4. **Prediction History**
- Tracks all predictions made during the session
- Shows timestamp, problem title, difficulty class, and rating
- Persists during the session using Streamlit session state

### User Workflow
1. Enter problem details in the form
2. Click "🔍 Predict Difficulty" button
3. View predicted difficulty class and rating
4. Check confidence scores
5. Review prediction in history tab
6. Make additional predictions or clear history

---

## 🎥 Demo Video


*Demo covers:*
- Brief project overview
- Application walkthrough
- Entering problem details
- Viewing predictions
- Understanding results
- Using prediction history

---

## 👨‍💻 Author Details

**Name:** Aditya Sharma
**Enrolment No:** 24115013
**GitHub:** [@adityapro3](https://github.com/adityapro3)  
**Repository:** [github.com/adityapro3/autojudge](https://github.com/adityapro3/autojudge)  
**Tech Stack:** Python, Scikit-learn, Streamlit, Pandas, TF-IDF

---

## 📄 License

This project is licensed under the MIT License.

---

