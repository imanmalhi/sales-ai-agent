import os
from os.path import join, dirname
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

os.environ["OPENAI_API_KEY"] = os.environ.get("API_KEY")

# App Framework
st.title("Sales AI Agent")
prompt = st.text_input(
    "I can help you create sales pitches. What do you want to write about?"
)

title_template = PromptTemplate(
    input_variables=["topic"],
    template="Write me a sales pitch title about {topic}",
)

script_template = PromptTemplate(
    input_variables=["title", "wikipedia_research"],
    template="Write me a sales pitch promotional email about this title: {title} while leveraging the following research: {wikipedia_research}",
)

# Memory
title_memory = ConversationBufferMemory(
    input_key="topic",
    memory_key="chat_history"
)
script_memory = ConversationBufferMemory(
    input_key="title",
    memory_key="chat_history"
)

# LLMs
llm = OpenAI(temperature=0.1)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key="title", memory=title_memory)
script_chain = LLMChain(
    llm=llm, prompt=script_template, verbose=True, output_key="script", memory=script_memory
)

wiki = WikipediaAPIWrapper()

# sequential_chain = SequentialChain(
#     chains=[title_chain, script_chain],
#     input_variables=["topic"],
#     output_variables=["title", "script"],
#     verbose=True,
# )

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Prompt Response
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    
    st.write(title)
    st.write(script)
    
    with st.expander("Title History"):
        st.info(title_memory.buffer)
        
    with st.expander("Script History"):
        st.info(script_memory.buffer)
        
    with st.expander("Wikipedia Research"):
        st.info(wiki_research)
