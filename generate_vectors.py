import os
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter




def load_tz_document(file_path: str) -> str:
    """Загружает содержимое ТЗ из файла."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def split_document(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Разбивает документ на чанки для лучшего поиска.
    
    Args:
        text: Текст документа
        chunk_size: Размер чанка в символах
        chunk_overlap: Перекрытие между чанками
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

def create_vector_store(documents: list, persist_directory: str = "./vector_db"):
    """
    Создает векторное хранилище из документов.
    
    Args:
        documents: Список текстовых чанков
        persist_directory: Директория для сохранения векторов
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': 'cpu'}
    )
    

    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"✓ Векторное хранилище создано в директории: {persist_directory}")
    print(f"✓ Обработано {len(documents)} чанков документа")
    
    return vectorstore

def main():
    """Основная функция для генерации векторов."""
    tz_file = "ТЗ_Телеграм_Бот_Кофейня.md"
    
    if not os.path.exists(tz_file):
        print(f"Ошибка: Файл {tz_file} не найден!")
        return
    
    print("Загрузка технического задания...")
    document_text = load_tz_document(tz_file)
    print(f"✓ Загружено {len(document_text)} символов")
    
    print("\n Разбиение документа на чанки...")
    chunks = split_document(document_text)
    print(f"Создано {len(chunks)} чанков")
    
    print("\nГенерация векторных представлений...")
    print("(Это может занять несколько минут при первом запуске)")
    
    vectorstore = create_vector_store(chunks)
    
    print("\nГотово! Векторная база данных создана.")
    print(f"Файлы векторов сохранены в: ./vector_db/")
    print("\nТеперь вы можете использовать rag_query.py для работы с RAG-системой.")

if __name__ == "__main__":
    main()

