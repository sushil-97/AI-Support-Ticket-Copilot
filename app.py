
import streamlit as st
import joblib

# Load trained models
vectorizer = joblib.load("category_vectorizer.pkl")
svm_model = joblib.load("category_model.pkl")

urgency_vectorizer = joblib.load("urgency_vectorizer.pkl")
urgency_svm = joblib.load("urgency_model.pkl")


# Priority mapping
priority_mapping = {
    "high": "P1",
    "medium": "P2",
    "low": "P3"
}

# Queue mapping
queue_mapping = {
    "billing": "Billing & Payments",
    "bug": "Technical Support",
    "refund": "Returns & Exchanges",
    "other": "General Support"
}

# Recommended actions
action_mapping = {
    "P1": "Escalate immediately and notify the technical team.",
    "P2": "Assign to the relevant support team for prompt review.",
    "P3": "Add to the normal support queue."
}

# Suggested responses
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

    # Category prediction
    category_tfidf = vectorizer.transform([ticket_text])
    category = svm_model.predict(category_tfidf)[0]

    # Urgency prediction
    urgency_tfidf = urgency_vectorizer.transform([ticket_text])
    urgency = urgency_svm.predict(urgency_tfidf)[0]

    # Routing
    priority = priority_mapping.get(urgency, "P3")
    queue = queue_mapping.get(category, "General Support")
    action = action_mapping.get(
        priority,
        "Review ticket manually."
    )

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


# Streamlit page
st.set_page_config(
    page_title="AI Support Copilot",
    page_icon="🎫",
    layout="centered"
)

st.title("🎫 AI Support Ticket Copilot")

st.write(
    "Enter a customer support ticket below. "
    "The system will classify, prioritize, route, "
    "and suggest a response."
)

ticket_text = st.text_area(
    "Customer Support Ticket",
    height=180,
    placeholder="Example: Our website is completely down and customers cannot log in."
)

if st.button("Analyze Ticket"):

    if not ticket_text.strip():

        st.warning("Please enter a support ticket.")

    else:

        result = support_copilot(ticket_text)

        st.subheader("Ticket Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Category",
                result["category"].title()
            )

            st.metric(
                "Priority",
                result["priority"]
            )

        with col2:
            st.metric(
                "Urgency",
                result["urgency"].title()
            )

            st.metric(
                "Recommended Queue",
                result["recommended_queue"]
            )

        st.subheader("Recommended Action")
        st.info(result["recommended_action"])

        st.subheader("Suggested Response")
        st.success(result["suggested_response"])
