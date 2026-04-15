from langchain.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise, helpful assistant. Answer the user's question
using ONLY the context provided below. If the answer is not contained in the
context, say "I don't have enough information to answer that."

Do not make up facts. Always cite the source filename and page number
at the end of your answer in this format: [Source: filename, page N]

Context:
{context}"""),
    ("human", "{question}"),
])