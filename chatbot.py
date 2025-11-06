from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
import time
from collections import deque

#load the env variables
load_dotenv()

#streamlit page setup
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered",
)
st.title("🗪 GenAI Chatbot")

#Initiate chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

#initiate throughput
if "request_times" not in st.session_state:
    st.session_state.request_times = deque(maxlen=10)

#Show chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#LLM initiate
llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
)

#Input box
user_prompt = st.chat_input("Ask your chat buddy 🖐")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user","content": user_prompt})

    start_time = time.time()

    response=llm.invoke(
        input = [{"role": "system", "content": "You are a helpful assistant"}, *st.session_state.chat_history]
    )
    assistant_response = response.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    end_time = time.time()
    latency = end_time - start_time

    current_time = time.time()
    st.session_state.request_times.append(current_time)

    #Throughput Calculation
    if(len(st.session_state.request_times)>1):
        time_window = st.session_state.request_times[-1] - st.session_state.request_times[0]
        throughput = len(st.session_state.request_times)/time_window if time_window>0 else 0
    else:
        throughput = 0

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    #LLM Evaluation
    print(f"Latency: {latency:.2f} seconds") #Latency
    print(f"Throughput: {throughput:.2f} req/sec") #Throughput



