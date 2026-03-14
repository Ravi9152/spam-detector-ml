import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page config
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# Title and description
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📧 Spam Detector</h1>", unsafe_allow_html=True)
st.write("Type any email or SMS message below and check if it's **Spam** or **Not Spam**.")

# Input box
message = st.text_area("✍️ Enter your message:", height=150)

# Button
if st.button("🔍 Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message first!")
    else:
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)
        result = "🚨 Spam" if prediction[0] == 1 else "✅ Not Spam"

        # Stylish output
        if prediction[0] == 1:
            st.error(f"Result: {result}")
        else:
            st.success(f"Result: {result}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
