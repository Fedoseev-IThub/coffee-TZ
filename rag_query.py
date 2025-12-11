

import os
from pathlib import Path


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def load_vector_store(persist_directory: str = "./vector_db"):
    """
    Загружает векторное хранилище из директории.
    
    Args:
        persist_directory: Директория с векторной базой данных
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Векторная база данных не найдена в {persist_directory}.\n"
            f"Сначала запустите generate_vectors.py для создания векторов."
        )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': 'cpu'}
    )
    
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vectorstore

def create_qa_chain(vectorstore, use_local_llm: bool = True):
    """
    Создает цепочку вопрос-ответ на основе векторного хранилища.
    
    Args:
        vectorstore: Векторное хранилище
        use_local_llm: Использовать локальную модель (Ollama) или OpenAI
    """

    prompt_template = """Используй следующую информацию из технического задания для ответа на вопрос.
Если в предоставленной информации нет ответа на вопрос, скажи об этом.

Контекст:
{context}

Вопрос: {question}

Ответ на русском языке:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    

    if use_local_llm:

        try:
            llm = Ollama(model="llama2", temperature=0.7)
        except Exception as e:
            print(f"Ошибка подключения к Ollama: {e}")
            print("Установите Ollama: https://ollama.ai/")
            print("Или используйте OpenAI, установив OPENAI_API_KEY")
            raise
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY не установлен. "
                "Установите переменную окружения или используйте локальную модель."
            )
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=api_key
        )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def query_rag(question: str, qa_chain):
    """
    Выполняет запрос к RAG-системе.
    
    Args:
        question: Вопрос пользователя
        qa_chain: Цепочка вопрос-ответ
    """
    print(f"\nВопрос: {question}")
    print("Поиск релевантной информации...")
    
    result = qa_chain({"query": question})
    
    print("\n" + "="*60)
    print("Ответ:")
    print("="*60)
    print(result["result"])
    
    print("\n" + "="*60)
    print("Использованные источники:")
    print("="*60)
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"\n[{i}] Фрагмент из ТЗ:")
        print("-" * 60)
        print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
        print("-" * 60)
    
    return result

def interactive_mode(qa_chain):
    """Интерактивный режим для задавания вопросов."""
    print("\n" + "="*60)
    print("RAG-система для работы с техническим заданием")
    print("="*60)
    print("Введите ваш вопрос о ТЗ (или 'выход' для завершения):\n")
    
    while True:
        question = input("> ").strip()
        
        if question.lower() in ['выход', 'exit', 'quit', 'q']:
            print("\n👋 До свидания!")
            break
        
        if not question:
            continue
        
        try:
            query_rag(question, qa_chain)
            print("\n" + "-"*60 + "\n")
        except Exception as e:
            print(f"Ошибка: {e}\n")

def main():
    """Основная функция."""
    import sys
    
    use_local = True
    if "--openai" in sys.argv:
        use_local = False
    
    try:
        print("Загрузка векторной базы данных...")
        vectorstore = load_vector_store()
        print("Векторная база данных загружена")
        
        print("\nСоздание RAG-цепочки...")
        qa_chain = create_qa_chain(vectorstore, use_local_llm=use_local)
        print("RAG-система готова к работе")
        
        if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
            question = " ".join([arg for arg in sys.argv[1:] if not arg.startswith("--")])
            query_rag(question, qa_chain)
        else:
            interactive_mode(qa_chain)
            
    except FileNotFoundError as e:
        print(f"{e}")
        print("\n💡 Сначала запустите: python generate_vectors.py")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()

