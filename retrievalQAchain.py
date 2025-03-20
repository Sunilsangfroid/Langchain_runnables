from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

loader = TextLoader("docs.txt") # Ensure that docs.txt is in the same directory as this script
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

retriever = vector_store.as_retriever()

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=retriever)

query = "What are the key takeways from the document?"
answer = qa_chain.run(query=query)

print("Answer: ",answer)
