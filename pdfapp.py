# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:08:40 2024

@author: yash
"""

## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## set up Streamlit 
st.title("Conversational RAG With PDF uplaods and chat history")
st.write("Upload Pdf's and chat with their content")

llm=ChatGroq(groq_api_key=groq_api_key,model_name='llama3-70b-8192')


uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
        ## Process uploaded  PDF's
if uploaded_files:
    
    
    documents=[]
    for uploaded_file in uploaded_files:
        
        temppdf=f"./temp.pdf"
        
        with open(temppdf,"wb") as file:
            
            file.write(uploaded_file.getvalue())
            
            file_name=uploaded_file.name
            

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
            
            
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )
            
            question_answer_chain=create_stuff_documents_chain(llm,prompt)
            rag_chain=create_retrieval_chain(retriever,question_answer_chain)
            
            output_parser=StrOutputParser()
            
            chain=prompt|llm|output_parser
            input_text=st.text_input("What question you have in mind?")
            
            if input_text:
                
                context = retriever.get_relevant_documents(input_text)
                formatted_context = "\n".join([doc.page_content for doc in context])
    
    # Adjusting the invocation of the chain
                st.write(chain.invoke({"context": formatted_context, "input": input_text}))
            
#            response=rag_chain.invoke({"input":"Tell me best phone camera ?"})
#            response

                        




