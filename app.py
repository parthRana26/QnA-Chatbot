import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory

# =====================
# LOAD ENV
# =====================
load_dotenv()

# =====================
# LLM
# =====================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# =====================
# VECTOR DB (RAG)
# =====================
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="db",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever()

# =====================
# PROMPT
# =====================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant and your name is RAGBOT."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}")
])

# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 My Chatbot")

# Initialize memory
if "history" not in st.session_state:
    st.session_state.history = InMemoryChatMessageHistory()

# Display chat history
for msg in st.session_state.history.messages:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

# =====================
# USER INPUT
# =====================
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.chat_message("user").write(user_input)

    # =====================
    # RAG: Retrieve context
    # =====================
    docs = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])

    # =====================
    # HYBRID LOGIC
    # =====================
    if context.strip():
        final_input = f"""
        Answer based on the context below.

        Context:
        {context}

        Question:
        {user_input}
        """
    else:
        final_input = user_input

    # =====================
    # PROMPT + LLM
    # =====================
    prompt_value = prompt.invoke({
        "user_input": final_input,
        "history": st.session_state.history.messages
    })

    response = llm.invoke(prompt_value)

    # =====================
    # DISPLAY RESPONSE
    # =====================
    st.chat_message("assistant").write(response.content)

    # =====================
    # SAVE MEMORY
    # =====================
    st.session_state.history.add_user_message(user_input)
    st.session_state.history.add_ai_message(response.content)