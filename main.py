from tempfile import NamedTemporaryFile
import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools_1 import ImageCaptionTool, ObjectDetectionTool
from PIL import Image

# Initialize tools only once
tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=1,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key="#ApiKeyOfYours",
    temperature=0.5,
    model_name="gpt-3.5-turbo"
)

agent = initialize_agent(
    agent="structured-chat-zero-shot-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory
)

st.title('Ask a question to image')
st.header('Please upload an image')
file = st.file_uploader("Upload an Image", type=["jpeg", "jpg", "png"])

if file:
    st.image(file, use_column_width=True)

    user_question = st.text_input('Ask question about image: ')

    if user_question and user_question.strip():
        with NamedTemporaryFile(dir='.') as f:
            f.write(file.getbuffer())
            image_path = f.name
            with st.spinner(text='In progress...'):
                response = agent.run(f'{user_question}, this is the image path: {image_path}')
                st.write(response)
