import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
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
# PROMPT
# =====================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Your name is Robust Artificial Genius. Created by Mr. Parth Rana."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}")
])

# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI Chatbot")

# =====================
# MEMORY INIT
# =====================
if "history" not in st.session_state:
    st.session_state.history = InMemoryChatMessageHistory()

# =====================
# DISPLAY CHAT HISTORY
# =====================
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
    # PROMPT + MEMORY
    # =====================
    prompt_value = prompt.invoke({
        "user_input": user_input,
        "history": st.session_state.history.messages
    })

    # =====================
    # LLM RESPONSE
    # =====================
    response = llm.invoke(prompt_value)

    # Show AI response
    st.chat_message("assistant").write(response.content)

    # =====================
    # SAVE MEMORY
    # =====================
    st.session_state.history.add_user_message(user_input)
    st.session_state.history.add_ai_message(response.content)
