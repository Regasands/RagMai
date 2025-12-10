from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

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




def split_documents(documents, chunk_size=500, chunk_overlap=150):
    """Разделяет документы на перекрывающиеся фрагменты"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(documents)


def create_vector_store(documents,  embeddings,  persist_directory="./chroma_db",):
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
