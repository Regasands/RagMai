# 1. Установите зависимости
# pip install langchain langchain-gigachat chromadb pypdf python-docx

import os
from langchain_gigachat import GigaChatEmbeddings, GigaChat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# 2. Настройка аутентификации
load_dotenv() 

# Или: credentials = "ваш_ключ"

# 3. Инициализация моделей
embeddings = GigaChatEmbeddings(
    credentials=os.getenv("GIGA_CHAT_KEY"),
    verify_ssl_certs=False,
    model="Embeddings"  # Или "EmbeddingsGigaR" для больших контекстов
)
print(os.getenv)
llm = GigaChat(
    credentials=os.getenv("GIGA_CHAT_KEY"),
    verify_ssl_certs=False,
    model="GigaChat",  # или "GigaChat-Plus"
    temperature=0.1,
    streaming=True
)

# 4. Загрузка и обработка документов
def load_documents(file_paths):
    """Загружает документы разных форматов"""
    documents = []
    
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            print(f"Формат {file_path} не поддерживается")
            continue
        
        documents.extend(loader.load())
    
    return documents

# 5. Разделение на чанки
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Разделяет документы на перекрывающиеся фрагменты"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(documents)

# 6. Создание векторной базы
def create_vector_store(documents, persist_directory="./chroma_db"):
    """Создает и сохраняет векторную базу"""
    # Разделяем документы
    chunks = split_documents(documents)
    
    # Создаем векторную базу
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    vector_store.persist()
    print(f"Создано {len(chunks)} чанков, сохранено в {persist_directory}")
    return vector_store

# 7. Кастомный промпт для RAG
prompt_template = """Ты — полезный ассистент, который отвечает на вопросы на основе предоставленного контекста.
Используй только информацию из контекста. Если ответа нет в контексте, скажи "В предоставленных документах нет информации по этому вопросу".

Контекст: {context}

Вопрос: {question}

Ответ:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 8. Основная функция RAG
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
        elif document_paths:
            # Создаем новую базу из документов
            print("Загрузка документов...")
            documents = load_documents(document_paths)
            self.vector_store = create_vector_store(documents)
        else:
            raise ValueError("Необходимо указать либо документы, либо путь к векторной базе")
        
        # Создаем цепочку RAG
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Берем 5 наиболее релевантных чанков
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
    rag_system = RAGSystem(document_paths=document_paths)
    
    # Вариант 2: Загрузка существующей базы
    # rag_system = RAGSystem(vector_store_path="./chroma_db")
    
    # Задаем вопрос
    question = "Что такое предел?"
    
    # Получаем ответ
    result = rag_system.ask(question)
    
    print(f"Вопрос: {question}")
    print(f"Ответ: {result['answer']}")
    print(f"Источники: {result['sources']}")
    print(f"Использовано чанков: {result['relevant_chunks']}")