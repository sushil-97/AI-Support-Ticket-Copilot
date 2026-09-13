import streamlit as st
import joblib

# -----------------------------
# Load Models
# -----------------------------
vectorizer = joblib.load("category_vectorizer.pkl")
svm_model = joblib.load("category_model.pkl")

urgency_vectorizer = joblib.load("urgency_vectorizer.pkl")
urgency_svm = joblib.load("urgency_model.pkl")

# -----------------------------
# Business Logic
# -----------------------------
priority_mapping = {
    "high": "P1",
    "medium": "P2",
    "low": "P3"
}

queue_mapping = {
    "billing": "Billing & Payments",
    "bug": "Technical Support",
    "refund": "Returns & Exchanges",
    "other": "General Support"
}

action_mapping = {
    "P1": "Escalate immediately and notify the technical team.",
    "P2": "Assign to the relevant support team for prompt review.",
    "P3": "Add to the normal support queue."
}

response_mapping = {
    "billing": (
        "Thank you for contacting support. "
        "We are reviewing your billing concern and will assist you "
        "with the payment or invoice issue."
    ),
    "bug": (
        "Thank you for reporting this technical issue. "
        "Our technical support team will investigate the problem "
        "and work on resolving it."
    ),
    "refund": (
        "Thank you for contacting us. "
        "We will review your return, exchange, or refund request "
        "and guide you through the next steps."
    ),
    "other": (
        "Thank you for contacting support. "
        "Your request has been received and will be reviewed "
        "by the appropriate team."
    )
}

def support_copilot(ticket_text):
    category_tfidf = vectorizer.transform([ticket_text])
    category = svm_model.predict(category_tfidf)[0]

    urgency_tfidf = urgency_vectorizer.transform([ticket_text])
    urgency = urgency_svm.predict(urgency_tfidf)[0]

    priority = priority_mapping.get(urgency, "P3")
    queue = queue_mapping.get(category, "General Support")
    action = action_mapping.get(priority, "Review ticket manually.")
    suggested_response = response_mapping.get(
        category,
        "Thank you for contacting support. We will review your request."
    )

    return {
        "category": category,
        "urgency": urgency,
        "priority": priority,
        "recommended_queue": queue,
        "recommended_action": action,
        "suggested_response": suggested_response
    }

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Support Copilot",
    page_icon="🎫",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 2.2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1f2937);
    margin-bottom: 2rem;
}

.hero h1 {
    margin: 0;
    font-size: 2.7rem;
    color: white;
}

.hero p {
    color: #d1d5db;
    font-size: 1.05rem;
    margin-top: 0.7rem;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}

.result-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    min-height: 115px;
}

.result-label {
    font-size: 0.85rem;
    opacity: 0.7;
    margin-bottom: 0.5rem;
}

.result-value {
    font-size: 1.35rem;
    font-weight: 700;
}

.info-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
}

.footer {
    text-align: center;
    opacity: 0.65;
    padding-top: 2rem;
    font-size: 0.85rem;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3rem;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🎫 Support Copilot")

    st.markdown("""
    ### About

    This application uses machine learning to analyze customer support tickets.

    **Models**
    - Category: TF-IDF + LinearSVC
    - Urgency: TF-IDF + LinearSVC

    **Outputs**
    - Category
    - Urgency
    - Priority
    - Support Queue
    - Recommended Action
    - Suggested Response
    """)

    st.divider()

    st.markdown("### Model Performance")
    st.metric("Category Accuracy", "~76%")
    st.metric("Urgency Accuracy", "~69%")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🎫 AI Support Ticket Copilot</h1>
    <p>
        Intelligent ticket classification, prioritization and routing
        powered by Natural Language Processing and Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Main Input Area
# -----------------------------
st.markdown(
    '<div class="section-title">Analyze a Support Ticket</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter a customer support request below and the system will "
    "automatically classify, prioritize and route it."
)

# -----------------------------
# Ticket Input
# -----------------------------

sample_ticket = (
    "Our production website is currently unavailable and "
    "customers are unable to log in. Please investigate urgently."
)

# Initialize the text area state
if "ticket_input" not in st.session_state:
    st.session_state.ticket_input = ""

# Function used by the sample button
def load_sample_ticket():
    st.session_state.ticket_input = sample_ticket

ticket_text = st.text_area(
    "Customer Support Ticket",
    height=190,
    placeholder=(
        "Example: Our production website is currently unavailable "
        "and customers are unable to log in. Please investigate urgently."
    ),
    key="ticket_input",
    label_visibility="collapsed"
)

col_a, col_b = st.columns([3, 1])

with col_a:
    analyze = st.button(
        "🔍 Analyze Ticket",
        type="primary",
        use_container_width=True
    )

with col_b:
    st.button(
        "Use Sample Ticket",
        use_container_width=True,
        on_click=load_sample_ticket
    )

# -----------------------------
# Prediction Output
# -----------------------------
if analyze:

    if not ticket_text.strip():
        st.warning("Please enter a support ticket before analyzing.")

    else:
        with st.spinner("Analyzing ticket..."):
            result = support_copilot(ticket_text)

        st.divider()

        st.markdown(
            '<div class="section-title">Ticket Analysis</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">CATEGORY</div>
                    <div class="result-value">
                        {result["category"].title()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">URGENCY</div>
                    <div class="result-value">
                        {result["urgency"].title()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">PRIORITY</div>
                    <div class="result-value">
                        {result["priority"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">ROUTING QUEUE</div>
                    <div class="result-value">
                        {result["recommended_queue"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div class="info-card">
                <strong>⚡ Recommended Action</strong>
                <br><br>
                {result["recommended_action"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="info-card">
                <strong>💬 Suggested Customer Response</strong>
                <br><br>
                {result["suggested_response"]}
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    Built with Python • Scikit-learn • TF-IDF • LinearSVC • Streamlit
</div>
""", unsafe_allow_html=True)
