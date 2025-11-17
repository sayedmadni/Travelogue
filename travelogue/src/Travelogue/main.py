# This is the main module for the Travelogue project everything should be called from this file. 
# This is the entry point for the application and the only place where we will have stremlit.
import streamlit as st
from Utilities.Ollama.LLM import inference_server_model

st.title("Travelogue")
st.write("This is a travelogue project to track my travels")

if st.button("Generate Travelogue"):
    st.write(inference_server_model("I want to go to the beach", "I want to go to the beach"))