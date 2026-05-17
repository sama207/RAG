import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="CV RAG Assistant",
    page_icon="📄",
    layout="centered",
)

st.title("📄 CV RAG Assistant")
st.write("Ask questions about the CVs in your RAG system.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")

    show_context = st.checkbox("Show retrieved context", value=False)
    st.write("Show context:", show_context)

    if st.button("Check backend health"):
        try:
            response = requests.get(f"{API_BASE_URL}/health")

            if response.status_code == 200:
                st.success("Backend is running ✅")
            else:
                st.error("Backend is not responding correctly ❌")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI backend ❌")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("context"):
            with st.expander("Retrieved CV Context"):
                st.text(message["context"])

question = st.chat_input("Ask a question about the CVs...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching CVs and generating answer..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={
                        "question": question,
                        "show_context": show_context,
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()

                    answer = data["answer"]
                    context = data.get("context")

                    st.markdown(answer)

                    if context:
                        with st.expander("Retrieved CV Context"):
                            st.text(context)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "context": context,
                    })

                else:
                    st.error(f"Backend error: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI. Run: uvicorn main:app --reload")

            except requests.exceptions.Timeout:
                st.error("The request took too long. Try again.")