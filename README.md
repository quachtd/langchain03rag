# langchain03chatbot

A chatbot with RAG and Tool Calling:
- LangChain 03
- LangGraph
- GUI with Chainlit

### Install
python3 -m venv .venv\
source .venv/bin/activate\
pip install -r requirements.txt

#### OpenAI Key
create an file .env
```
OPENAI_API_KEY='your key here'
```
### Run
chainlit run chatbot2_toolscalling_chatlit.py -w

### Try prompts
1. What is Task Decomposition?
2. Can you look up some common ways of doing it?
3. what is "it" here?