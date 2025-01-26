import streamlit as st
import requests

# Define Streamlit page configuration
st.set_page_config(page_title="Financial Report Insights", layout="wide")

# Title of the Streamlit app
st.title("Financial Report Insights")

# Input form for the user to ask questions
st.subheader("Ask your question related to financial reports:")

# Create an input box for user queries
user_query = st.text_input("Enter your question:")

if user_query:
    # Make an API call to the RAG question-answering service (adjust the endpoint as needed)
    response = requests.post(
        'http://localhost:8000/query',  # Replace with your app's URL if necessary
        json={"query": user_query}
    )

    if response.status_code == 200:
        answer = response.json().get("answer", "No answer found.")
        st.write(f"Answer: {answer}")
    else:
        st.error("Error querying the server.")
