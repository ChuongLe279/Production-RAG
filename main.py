from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
#from langchain_community.document_loaders import TextLoader
from bs4.filter import SoupStrainer
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import JsonOutputParser
# LLM
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

### INDEXING
# Load Documents: load every pdf in data
data_path = Path("./data")

docs = []
for pdf_path in data_path.glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_path))
    docs.extend(loader.load())

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splits = text_splitter.split_documents(docs)

#print(f"spits: {splits}")


# Embed
vector_store = Chroma.from_documents(documents=splits[:20],
                                     embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"))
retriver = vector_store.as_retriever()

# Decomposition
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Return ONLY a JSON array of strings, no other text. The answer to the question is located in the vector store because
this a a rag pipeline.
Example output: ["question 1", "question 2", "question 3"]
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)

# Decomposition chain
question = "What is the name of authors the work on those scientific papers?"
generate_queries_decomposition = (
    {"question": RunnablePassthrough()}
    | prompt_decomposition
    | gemini
    | JsonOutputParser()
)
decomposed_questions = generate_queries_decomposition.invoke(question)

"""
print(f"decomposed questions: {decomposed_questions}")
decomposed questions: ["How to find authors of a scientific paper"
, "Scientific paper databases for author information"
, "Tool to extract author names from multiple research papers"]
"""

# Multi-querry 
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_multiple_querries = ChatPromptTemplate.from_template(template)
multi_querries_questions = {}
for idx, ques in enumerate(decomposed_questions):
    generate_multi_querries = (
        {"question": RunnablePassthrough()}
        | prompt_multiple_querries
        | gemini
        | StrOutputParser()
    )
    multi_querries_questions[idx] = generate_queries_decomposition.invoke(ques)
"""
print(f"Multi querries; {multi_querries_questions}")
Multi querries; {
    0: ['Where are authors listed on a scientific paper?', 'How to find authors of a research paper using Google Scholar?', 'Identify authors of a scientific paper by DOI or title'], 
    1: ['tools to identify authors from scientific paper titles or DOIs', 'software to find authors from scientific paper DOI', 'how to determine author from scientific paper title'], 
    2: ['List of academic databases for scientific publications', 'How to find authors using scientific publication databases', 'Best databases for tracking scientific authors and their publications']}
"""

# Normal RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""
prompt_normal_rag = ChatPromptTemplate.from_template(template)


    
