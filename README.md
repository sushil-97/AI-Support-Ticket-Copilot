# AI-Support-Ticket-Copilot Architecture

Support Ticket
      ↓
Text Preprocessing
      ↓
TF-IDF
      ↓
 ┌──────────────┬──────────────┐
 │ Category SVM │ Urgency SVM  │
 └──────────────┴──────────────┘
      ↓
Priority + Queue Routing
      ↓
Recommended Action
      ↓
Suggested Response
      ↓
Streamlit Web App
