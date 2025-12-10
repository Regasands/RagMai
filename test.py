# 1. Установите зависимости
# pip install langchain langchain-gigachat chromadb pypdf python-docx

import os
from langchain_gigachat import GigaChatEmbeddings, GigaChat

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from dotenv import load_dotenv


from tools.document import load_documents, create_vector_store
from promthub.base import PROMPT


load_dotenv() 


embeddings = GigaChatEmbeddings(
    credentials=os.getenv("GIGA_CHAT_KEY"),
    verify_ssl_certs=False,
    model="Embeddings" 
)


llm = GigaChat(
    credentials=os.getenv("GIGA_CHAT_KEY"),
    verify_ssl_certs=False,
    model="GigaChat", 
    temperature=0.1,
    streaming=True
)



class RAGSystem:
    def __init__(self, document_paths=None, vector_store_path=None):
        self.embeddings = embeddings
        self.llm = llm
        
        if vector_store_path and os.path.exists(vector_store_path):
            # Загружаем существующую базу
            self.vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=self.embeddings
            )

            print(f"Загружена векторная база из {vector_store_path}")

        # нужно только в начале для понимания длля первичяные инизиалции далее можно удет решить этот костыь с помозщью проверки
        elif document_paths:
            # Создаем новую базу из документов
            print("Загрузка документов...")
            documents = load_documents(document_paths)
            self.vector_store = create_vector_store(documents, embeddings)

        else:
            raise ValueError("Необходимо указать либо документы, либо путь к векторной базе")
        
        # Создаем цепочку RAG
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="mmr",  # MMR лучше работает с разнообразными результатами
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,  # Сначала берем больше документов
                    "lambda_mult": 0.7 # Баланс между релевантностью и разнообразием
                }
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def ask(self, question):
        """Задаем вопрос системе"""
        result = self.qa_chain.invoke({"query": question})
        
        # Форматируем ответ
        answer = result["result"]
        sources = list(set([doc.metadata.get("source", "Unknown") 
                          for doc in result["source_documents"]]))
        
        return {
            "answer": answer,
            "sources": sources,
            "relevant_chunks": len(result["source_documents"])
        }

# 9. Использование системы
if __name__ == "__main__":
    # Вариант 1: Создание с нуля из документов
    document_paths = [
        "book1.pdf",
    ]
    
    # Инициализация RAG системы
    # rag_system = RAGSystem(document_paths=document_paths)
    
    # # Вариант 2: Загрузка существующей базы
    rag_system = RAGSystem(vector_store_path="./chroma_db")
    
    # Задаем вопрос
    question = "Определние предал по коши?"
    
    # Получаем ответ
    result = rag_system.ask(question)
    
    print(f"Вопрос: {question}")
    print(f"Ответ: {result['answer']}")
    print(f"Источники: {result['sources']}")
    print(f"Использовано чанков: {result['relevant_chunks']}")