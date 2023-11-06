import os
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

load_dotenv() # loads my openai API key
print(os.environ['OPENAI_API_KEY'])

# load the document as before
loader = PyPDFLoader('./docs/attention.pdf')
documents = loader.load()
# we split the data into chunks of 1,000 characters, with an overlap
# of 200 characters between the chunks, which helps to give better results
# and contain the context of the information between chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(documents)
# we create our vectorDB, using the OpenAIEmbeddings tranformer to create
# embeddings from our text chunks. We set all the db information to be stored
# inside the ./data directory, so it doesn't clutter up our source files

vectordb = Chroma.from_documents(
documents,
embedding=OpenAIEmbeddings(),
persist_directory='./data'
)
vectordb.persist()

# chain = load_qa_chain(llm=OpenAI(), verbose=True)
qa_chain = RetrievalQA.from_chain_type(
llm=OpenAI(),
retriever=vectordb.as_retriever(search_kwargs={'k': 4}),
return_source_documents=True,
)

query = "Who are authors of this paper?"

response = qa_chain({"query": query})
# response, source_docs = qa_chain.run(query)

print(f"\nQuery: {response['query']}")
print(f"Response:{response['result']}")
print(f"Source Documents: {response['source_documents']}")