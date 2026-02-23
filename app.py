# ##  RAG Q&A converrsation with PDF including Chat History

# import streamlit as st
# # from langchain.chains import create_history_aware_retriver,create_retrieval_chain
# # from langchain.chains.combine_documents import create_stuff_documents_chain
# # from langchain_chroma import Chroma
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
# from langchain_core.runnables import RunnableWithMessageHistory
# from langchain_core.runnables import RunnablePassthrough,RunnableLambda
# # MPH is for defining the session key and what kind of session histor
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import Chroma
# import os
# from dotenv import load_dotenv
# load_dotenv()

# os.environ['HF_TOKEN']=
# embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ## Set up streamlit app

# st.title("Conversational RAG with PDF uploads and chat history")
# st.write("Upload's PDF and Chat with your Content")


# ## INput the api key
# api_key=

# ## Check if groq api key provided
# if api_key:
#     llm=ChatGroq(groq_api_key=api_key,model_name="openai/gpt-oss-120b")
# ## Chat interface
#     session_id=st.text_input("Session ID",value="default_session")

#     ## statefully manage chat history

#     if 'store' not in st.session_state:
#         st.session_state.store={}

#     uploaded_files=st.file_uploader("Choose A pdf file",type="pdf",accept_multiple_files=True)
#     ## Process uploaded file
#     if uploaded_files:
#         documents=[]
#         for uploaded_file in uploaded_files:
#             ## first we will save in local (temperorly)
#             temppdf=f"./temp.pdf"
#             with open(temppdf,'wb') as file:
#                 file.write(uploaded_file.getvalue())
#                 file_name=uploaded_file.name

#                 ## In short we are opening that file which is uploaded and reading all the value and reading all name
            
#             loader=PyPDFLoader(temppdf) ##loading whatever pdf file we have
#             docs=loader.load()
#             documents.extend(docs)  #3 appending all the docs

#         ##Split and embedding  for documents

#         text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
#         splits=text_splitter.split_documents(documents)
#         vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
#         retriever=vectorstore.as_retriever()

#         ##New prompt for sysytem
    
#         contextualize_q_system_prompt=(
#             "Given a chat history and the latest user question "
#             "which might reference context in the chat history"
#             "formulate a standalone question which can be understood"
#             "without the chat history Do not answer the question"
#             "just reformulate it if needed and otherwise return it as is"
#         )

#         contextualize_q_prompt=ChatPromptTemplate.from_messages(
#             [
#                 ("system",contextualize_q_system_prompt),
#                 MessagesPlaceholder("chat_history"),  # placeholder
#                 ("human","{input}")
#             ]
#         )
        
#         parser=StrOutputParser()
#         # history_aware_retriver=create_history_aware_retriver(llm,retriver,contextualize_q_promot)
#         # history_aware_retriever =(llm|retriever|parser|contextualize_q_prompt)
#         # history_aware_retriever =(contextualize_q_prompt|llm|retriever|parser)
#         history_aware_retriever = (contextualize_q_prompt| llm| StrOutputParser()| retriever)

#         ## Answerr question prompt

#         system_prompt=(
#             "You are an assitant for question-answer tasks."
#             "Use the following pieces of retrieved context to answer"
#             "the question. If you don't know the answer, say that you"
#             "don't know Use three sentences maximum and keep the"
#             "answer concise"
#             "\n\n"
#             "{context}"
#         )
#         # qa_prompt=ChatPromptTemplate.from_messages(
#         #     [
#         #         ("system",system_prompt),
#         #         MessagesPlaceholder("chat_history"),
#         #         ("human","{input}")
#         #     ]
#         # )
#         qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}")
#     ]
# )

#         ## this is up to you how you want to do
#         # question_answer_chain=create_stuff_documents_chain(llm|qa_prompt)
#         question_answer_chain=qa_prompt|llm|parser
#         # rag_chain=create_retrievel_chain(history_aware_retriever,question_answer_chain)
#         # rag_chain=history_aware_retriever|question_answer_chain
#         from langchain_core.runnables import RunnablePassthrough

#         rag_chain = (
#                 {
#                 "context": history_aware_retriever,   # docs go here
#                 "input": RunnablePassthrough(),       # user question
#                 # "chat_history": RunnablePassthrough()
#                  }
#                     | question_answer_chain)

#     #     rag_retriver_chain = (
#     #     {
#     #         "context": history_aware_retriever,
#     #         "input": RunnablePassthrough(),
#     #         "chat_history": RunnablePassthrough()
#     #     }
#     #     | question_answer_chain  # qa_chain = qa_prompt | llm | StrOutputParser()
#     # )
#         def get_session_history(sessipn:str) ->BaseChatMessageHistory: ## when we call session it should have/give base chat hist
#             if session_id not in st.session_state.store: # sessioin_id's will auto store in it
#                 st.session_state.store[session_id]=ChatMessageHistory()  # it will store all the chat which will happen and the given and give it to BCMH
#             return st.session_state.store[session_id] ## wil return the chat message

#         conversational_rag_chain=RunnableWithMessageHistory(
#             rag_chain,get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer"
#         )

#         user_input=st.text_input("Enter Your Query :")
#         if user_input:
#             session_history=get_session_history(session_id)
#             response=conversational_rag_chain.invoke(
#                 {"input":user_input},
#                 config={
#                     "configurable":{"session_id":session_id}
#                 }, #construct a key "abc123" in 'store
#             )

#             st.write(st.session_state.store)
#             # st.success("Assitant :",response)
#             st.markdown("**Assistant:**")
#             st.write(response)
#             st.write("Chat History : ",session_history.messages)

# else:
#     st.warning("Please enter Groq API key")




    


##  RAG Q&A converrsation with PDF including Chat History

import streamlit as st
# from langchain.chains import create_history_aware_retriver,create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
# MPH is for defining the session key and what kind of session histor
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="📚",
    layout="wide",
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.stTextInput > div > div > input {
    border-radius: 12px;
    padding: 12px;
    background-color: #1e293b;
    color: white;
    border: 1px solid #334155;
}

.stFileUploader {
    background-color: #1e293b;
    border-radius: 12px;
    padding: 15px;
}

.chat-bubble-user {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    padding: 12px 18px;
    border-radius: 18px;
    color: white;
    margin-bottom: 8px;
    max-width: 70%;
}

.chat-bubble-bot {
    background: #1e293b;
    padding: 12px 18px;
    border-radius: 18px;
    color: #e2e8f0;
    margin-bottom: 8px;
    max-width: 70%;
    border: 1px solid #334155;
}

.section-card {
    background: rgba(30, 41, 59, 0.6);
    padding: 20px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
    border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)
## Set up streamlit app

st.markdown("""
<h1 style='text-align: center; font-size: 2.8rem;
background: linear-gradient(90deg,#3b82f6,#22d3ee);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;'>
📚 AI PDF Research Assistant
</h1>
<p style='text-align: center; color: #94a3b8; font-size: 1.1rem;'>
Upload one or more PDF documents and ask contextual questions.
The system retrieves relevant content and answers intelligently.
</p>
<hr style='border: 1px solid #334155; margin-top:20px;'>
""", unsafe_allow_html=True)


## INput the api key
api_key=os.getenv("GROQ_API_KEY")

## Check if groq api key provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="openai/gpt-oss-120b")
## Chat interface


    ## statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store={}
        

    # uploaded_files=st.file_uploader("Choose A pdf file",type="pdf",accept_multiple_files=True)
    col1, col2 = st.columns([1,1])

    with col1:

        session_id = st.text_input("🔑 Session ID", value="default_session")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        uploaded_files = st.file_uploader(
            "📄 Upload PDF Files",
            type="pdf",
            accept_multiple_files=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    ## Process uploaded file
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            ## first we will save in local (temperorly)
            temppdf=f"./temp.pdf"
            with open(temppdf,'wb') as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

                ## In short we are opening that file which is uploaded and reading all the value and reading all name
            
            loader=PyPDFLoader(temppdf,extraction_mode="plain") ##loading whatever pdf file we have
            docs=loader.load()
            documents.extend(docs)  #3 appending all the docs

        ##Split and embedding  for documents

        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorstore.as_retriever()

        ##New prompt for sysytem
    
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question "
            "which might reference context in the chat history"
            "formulate a standalone question which can be understood"
            "without the chat history Do not answer the question"
            "just reformulate it if needed and otherwise return it as is"
        )

        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),  # placeholder
                ("human","{input}")
            ]
        )
        
        parser=StrOutputParser()
        # history_aware_retriver=create_history_aware_retriver(llm,retriver,contextualize_q_promot)
        # history_aware_retriever =(llm|retriever|parser|contextualize_q_prompt)
        # history_aware_retriever =(contextualize_q_prompt|llm|retriever|parser)
        history_aware_retriever = (contextualize_q_prompt| llm| StrOutputParser()| retriever)

        ## Answerr question prompt

        system_prompt=(
            "You are an assitant for question-answer tasks."
            "Use the following pieces of retrieved context to answer"
            "the question. If you don't know the answer, say that you"
            "don't know Use maximum sentence which say and"
            "keep the answer according to user demand "
            "if not demand then keep it short"
            "\n\n"
            "{context}"
        )
        # qa_prompt=ChatPromptTemplate.from_messages(
        #     [
        #         ("system",system_prompt),
        #         MessagesPlaceholder("chat_history"),
        #         ("human","{input}")
        #     ]
        # )
        qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

        ## this is up to you how you want to do
        # question_answer_chain=create_stuff_documents_chain(llm|qa_prompt)
        question_answer_chain=qa_prompt|llm|parser
        # rag_chain=create_retrievel_chain(history_aware_retriever,question_answer_chain)
        # rag_chain=history_aware_retriever|question_answer_chain
        from langchain_core.runnables import RunnablePassthrough

        rag_chain = (
                {
                "context": history_aware_retriever,   # docs go here
                "input": RunnablePassthrough(),       # user question
                # "chat_history": RunnablePassthrough()
                 }
                    | question_answer_chain)

    #     rag_retriver_chain = (
    #     {
    #         "context": history_aware_retriever,
    #         "input": RunnablePassthrough(),
    #         "chat_history": RunnablePassthrough()
    #     }
    #     | question_answer_chain  # qa_chain = qa_prompt | llm | StrOutputParser()
    # )
        def get_session_history(sessipn:str) ->BaseChatMessageHistory: ## when we call session it should have/give base chat hist
            if session_id not in st.session_state.store: # sessioin_id's will auto store in it
                st.session_state.store[session_id]=ChatMessageHistory()  # it will store all the chat which will happen and the given and give it to BCMH
            return st.session_state.store[session_id] ## wil return the chat message

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("💬 Ask something about your PDF")
        if user_input:
            session_history = get_session_history(session_id)

            with st.spinner("🔎 Thinking through your documents..."):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
        )

            st.markdown("<hr style='border: 1px solid #334155;'>", unsafe_allow_html=True)
# Display chat history in modern bubble style
            for msg in session_history.messages:
                if msg.type == "human":
                    st.markdown(f"<div class='chat-bubble-user'>{msg.content}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-bubble-bot'>{msg.content}</div>", unsafe_allow_html=True)

# Display latest response
            st.markdown(f"<div class='chat-bubble-bot'>{response}</div>", unsafe_allow_html=True)

else:
    st.warning("Please enter Groq API key")




    