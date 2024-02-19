# 1. Load raw documents using CSVLoader


from dotenv import load_dotenv
file_path = "conference_session_info.csv"
from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path=file_path)
docs = loader.load()


load_dotenv()


# 2. Split the raw documents into chunks
# RecursiveCharacterTextSplitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.




from langchain.text_splitter import RecursiveCharacterTextSplitter
chunk_size = 256
chunk_overlap = 32
r_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    add_start_index = True
)
pages = r_text_splitter.split_documents(docs)




# 3. Convert the chunks into embeddings and store them in ChromaDB
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
persist_directory = 'persist_chroma'
vectordb = Chroma.from_documents(
    documents=pages,
    embedding=embedding,
    persist_directory=persist_directory
)




# RetrievalQA
# The chain is constructed using (now legacy) RetrievalQA module. This chain first does a retrieval step to fetch relevant documents, then passes those documents into an LLM to generate a response.




from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import langchain
langchain.verbose = True




# llm_name = "gpt-3.5-turbo"
# llm = ChatOpenAI(model_name=llm_name, temperature=1)
llm = ChatOpenAI()
qa_chain_default = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    return_source_documents=True
)
question = "What is said about The City of Orange Valley?"
result = qa_chain_default.invoke({"query" : question})
print(result)




def pretty_print(text, words_per_line=15):
  words = text.split()
  for i in range(0, len(words), words_per_line):
    line = ' '.join(words[i:i+words_per_line])
    print(line)




pretty_print(result.get('result'))
