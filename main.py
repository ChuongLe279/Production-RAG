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
from langchain.load import dumps, loads 
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
def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results
multi_querries_questions = {}

generate_multi_querries = (
    {"question": RunnablePassthrough()}
    | prompt_multiple_querries
    | gemini
    | StrOutputParser()
)


for idx, ques in enumerate(decomposed_questions):
    multi_querries_questions[idx] = generate_multi_querries.invoke(ques)

"""
print(f"Multi querries; {multi_querries_questions}")
Multi querries; {
    0: ['Where are authors listed on a scientific paper?', 'How to find authors of a research paper using Google Scholar?', 'Identify authors of a scientific paper by DOI or title'], 
    1: ['tools to identify authors from scientific paper titles or DOIs', 'software to find authors from scientific paper DOI', 'how to determine author from scientific paper title'], 
    2: ['List of academic databases for scientific publications', 'How to find authors using scientific publication databases', 'Best databases for tracking scientific authors and their publications']}
"""

retrieved_docs = {}
# Fusion based on multi-queries context
for idx, queries in multi_querries_questions.items():
    query_list = [q.strip() for q in queries.split("\n") if q.strip()]
    docs = retriver.map().invoke(query_list)
    fused = reciprocal_rank_fusion(docs)
    retrieved_docs[idx] = fused[:2] # Take top 2 relevant documents

# Decomposed questions's answer based on multi queries context
template = """Return string only, no other text. The answer to the question is located in the vector store because
this a a rag pipeline.
Example output: answer.
Answer the question based only on the following context:
{context}

Question: {question}
"""
decomposed_questions_answer = []
decomposed_questions_prompt = ChatPromptTemplate.from_template(template)
for idx, question in enumerate(decomposed_questions):
    decomposed_questions_rag_chain = (
        {
            "context": retrieved_docs[idx],
            "question": RunnablePassthrough()
        }
        | decomposed_questions_prompt
        | gemini
        | StrOutputParser()
    )
    answer = decomposed_questions_rag_chain.invoke(question)  
    decomposed_questions_answer.append(answer) 

# Format q&a pairs (decomposed questions) and final answer
def format_qa(questions, answers):
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()
final_context = format_qa(decomposed_questions, decomposed_questions_answer)

# Final answer
template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""
final_prompt = ChatPromptTemplate.from_template(template)
final_rag_chain = (
    {
        "context": final_context,
        "question" : RunnablePassthrough()
    }
    | final_prompt
    | gemini
    | StrOutputParser()
)
final_response = final_rag_chain.invoke(question)
print(f"Final response: {final_response}")

