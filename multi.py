import os
import sys
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader

load_dotenv() # loads my openai API key
# print(os.environ['OPENAI_API_KEY'])

# load the document as before
# loader = PyPDFLoader('./docs/attention.pdf')
# documents = loader.load()

documents = []
for file in os.listdir('docs'):
    if file.endswith('.pdf'):
        pdf_path = './docs/' + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = './docs/' + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = './docs/' + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)
# we create our vectorDB, using the OpenAIEmbeddings tranformer to create
# embeddings from our text chunks. We set all the db information to be stored
# inside the ./data directory, so it doesn't clutter up our source files
vectordb = Chroma.from_documents(
    chunked_documents,
    embedding=OpenAIEmbeddings(),
    persist_directory='./data'
)
vectordb.persist()

qa_chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(),
    vectordb.as_retriever(search_kwargs={'k': 4}),
    return_source_documents=True
)

chat_history = []
while True:
    # this prints to the terminal, and waits to accept an input from the user
    query = input('Prompt (type "exit" to quit): ')
    # give us a way to exit the script
    if query == "exit" or query == "quit" or query == "q":
        print('Bye.')
        sys.exit()
    # we pass in the query to the LLM, and print out the response. As well as
    # our query, the context of semantically relevant information from our
    # vector store will be passed in, as well as list of our chat history
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'])
    # we build up the chat_history list, based on our question and response
    # from the LLM, and the script then returns to the start of the loop
    # and is again ready to accept user input.
    chat_history.append((query, result['answer']))