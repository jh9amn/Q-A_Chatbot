import streamlit as st
from utils.loaders import load_webpage
from utils.vectorstore import create_vectorstore
from utils.rag_chain import create_rag_chain


DEFAULT_STATE = {
    "rag_chain": None,
    "chat_history": [],
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


st.title("💬 Conversational RAG Chatbot")

url = st.text_input("Enter the URL of the webpage to load:")

if st.button("Process"):
    if not url:
        st.error("Please enter a valid URL.")
    else:
        with st.spinner("Loading and processing the webpage..."):
            docs = load_webpage(url)

            if not docs:
                st.error("No content found on the webpage.")
            else:
                vectordb = create_vectorstore(docs)
                st.session_state["rag_chain"] = create_rag_chain(vectordb)
                st.success("Webpage processed successfully!")
                
                
# ------------------ QUESTION INPUT ------------------
if st.session_state["rag_chain"]:
    st.divider()
    st.subheader("Ask a Question")

    user_question = st.text_input("Your question")

    if st.button("Get Answer"):
        if not user_question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                response = st.session_state["rag_chain"](
                    {"question": user_question}
                )

                st.markdown("### Answer")
                st.write(response["answer"])

                # Chat history
                st.markdown("### Chat History")
                for i, msg in enumerate(response["chat_history"]):
                    role = "User" if i % 2 == 0 else "Bot"
                    st.write(f"**{role}:** {msg.content}")