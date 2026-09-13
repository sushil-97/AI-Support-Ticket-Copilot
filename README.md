# AI-Support-Ticket-Copilot Architecture

# 🎫 AI Support Ticket Copilot

An end-to-end Machine Learning application that automatically analyzes customer support tickets, predicts their category and urgency, assigns a priority and support queue, recommends an action, and generates a suggested response.

The project is deployed as an interactive web application using Streamlit.

## 🚀 Live Demo

👉 [Try the AI Support Ticket Copilot](https://ai-support-ticket-copilot-fh6e6l6sghmyewpdwfzczn.streamlit.app/)

---

## 📌 Project Overview

Customer support teams receive large numbers of tickets that need to be classified, prioritized, and routed to the appropriate team.

This project demonstrates how Machine Learning can assist with this workflow.

The application takes a customer support ticket as input and produces:

- Ticket Category
- Urgency Level
- Priority
- Recommended Support Queue
- Recommended Action
- Suggested Customer Response

---

## 🧠 Machine Learning Approach

Two separate text classification models are used.

### 1. Category Classification

Predicts one of the following categories:

- Billing
- Bug
- Refund
- Other

**Model:** LinearSVC  
**Text Representation:** TF-IDF with unigrams and bigrams

Category model accuracy: **~76%**

### 2. Urgency Classification

Predicts:

- High
- Medium
- Low

**Model:** LinearSVC  
**Text Representation:** TF-IDF with unigrams and bigrams

Urgency model accuracy: **~69%**

---

## ⚙️ Application Workflow

```text
Customer Support Ticket
          |
          v
     Text Input
          |
          v
       TF-IDF
          |
     +----+----+
     |         |
     v         v
 Category   Urgency
 LinearSVC  LinearSVC
     |         |
     +----+----+
          |
          v
   Business Rules
          |
          v
 Priority + Queue
          |
          v
 Recommended Action
          |
          v
 Suggested Response
          |
          v
   Streamlit Web App
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- LinearSVC
- Joblib
- Streamlit
- Google Colab
- GitHub

---

## 📂 Project Structure

```text
AI-Support-Copilot/
│
├── app.py
├── category_model.pkl
├── category_vectorizer.pkl
├── urgency_model.pkl
├── urgency_vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## 💡 Example

### Input

```text
Our website is completely down and customers cannot log in.
```

### Output

The application predicts the ticket's:

```text
Category
Urgency
Priority
Recommended Queue
Recommended Action
Suggested Response
```

---

## 📊 Model Performance

| Model | Algorithm | Accuracy |
|---|---|---:|
| Category Classification | TF-IDF + LinearSVC | ~76% |
| Urgency Classification | TF-IDF + LinearSVC | ~69% |

The results also highlight the challenges of classifying real-world support tickets, including overlapping categories and noisy labels.

---

## 🔮 Future Improvements

Potential improvements include:

- Transformer-based models such as BERT
- Confidence scores and manual-review thresholds
- Better handling of multilingual tickets
- More granular ticket categories
- LLM-generated contextual responses
- Retrieval-Augmented Generation (RAG) using support documentation
- Model monitoring and feedback collection

---

## 👤 Author

**Sushil Pillay**

Machine Learning / AI Portfolio Project


