import asyncio
import os
from typing import List, Dict, TypedDict, Optional, Tuple
from pydantic import BaseModel, Field

import faiss
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from tqdm import tqdm

# Загрузка переменных окружения
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
BASE_URL = "https://ai-for-finance-hack.up.railway.app/"


# LLM_MODEL = 'openrouter/mistralai/mistral-small-3.2-24b-instruct'
# LLM_MODEL = 'meta-llama/llama-3-70b-instruct'
# LLM_MODEL = 'openrouter/meta-llama/llama-3-70b-instruct'
# LLM_MODEL = 'openrouter/x-ai/grok-3-mini'
LLM_MODEL = 'openrouter/google/gemma-3-27b-it'

EMBEDDER_MODEL = 'text-embedding-ada-002'
# EMBEDDER_MODEL = 'text-embedding-3-small'


MAX_EMBEDDING_TOKENS = 300_000


class DocumentSummary(BaseModel):
    reflections: str = Field(description="Твои размышления о том, есть ли в документе полезная информация для вопроса пользователя")
    necessity: bool = Field(description="Необходимость использования документа для ответа на запрос")
    final_summary: str = Field(description="Итоговое краткое содержание документа относительно запроса")

class DocumentAnswer(BaseModel):
    context_reasoning: str
    insufficient_context: bool
    answer_reasoning: str
    missing_information: List[str]
    answer: str


def count_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    """Возвращает количество токенов в текстовой строке."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Define state for application
class State(TypedDict):
    question: str
    context_reasoning: str
    insufficient_context: bool
    missing_information: List[str]
    answer_reasoning: str
    context: list[tuple[Document, float]]
    answer: str

class FinancialRAG:
    SYSTEM_PROMPT = """
Ты являешься финансовым экспертом. Твоя основная функция - точно отвечать на вопросы пользователей, **строго основываясь на предоставленных фрагментах контекста**.  
Точно соблюдай все правила форматирования и вывода.

## **ВОПРОС**
```
{question}
```

## **КОНТЕКСТ**
{context_str}
  > Каждый фрагмент имеет следующий формат:  
  > `ДОКУМЕНТ: <id> \n<текст фрагмента>`

## **ЦЕЛЬ**

Генерировать исчерпывающий, хорошо структурированный ответ на запрос пользователя, основанный **только** на приведенных фрагментах.  
Если информация отсутствует или недостаточна, прямо скажи об этом.

## **ПРАВИЛА**

1. **Только контекст:**  
   - Используй *только* информацию, явно содержащуюся в контексте и отвечающую на вопрос пользователя. Никаких догадок и предположений. 
   - Твои собственные знания должны использоваться ТОЛЬКО для того, чтобы свободно формулировать предложения и связывать идеи в последовательное повествование, а не для предоставления какой-либо внешней информации.

2. **Язык:**  
   Отвечай на том же языке, что и на вопрос.  

3. **Стиль:**  
   - Ответ должен точно отвечать на запрос пользователя. Не добавляй в ответ факты, которые никак не связаны с вопросом пользователя. 
   - Сохраняй нить повествования, не перепрыгивай с факта на факт, а послодовательно отвечай на вопрос пользователя.
   - В ответе ДОЛЖЕН быть использован формат Markdown для большей ясности и структуры (например, заголовки, выделенный жирным шрифтом текст, маркеры).
   - НЕ НУЖНО в ответе указывать из какого документа взята информация.
   - НЕ НУЖНО в ответе повторять вопрос пользователя. Начни сразу отвечать на вопрос.

4. **Тон:**
   Поддерживай профессиональный, дружелюбный и естественный тон при ответе на вопрос пользователя.

5. **Недостаточный контекст:**  
   Если ответ не может быть определен по фрагментам, установи `"insufficient_context": true` и укажи в `"missing_information"` те вопросы, на которые нужно найти информацию в базе знаний, чтобы дать более точный ответ на вопрос.

6. **Формат вывода:**  
   Точно следуй схеме JSON, приведенной ниже. Никакого дополнительного текста, пояснений или форматирования. 

```json
{{
  "context_reasoning": "<Твои КОРОТКИЕ рассуждения о том, достаточно ли информации в контексте для того, чтобы ответить на вопрос пользоваетя>",
  "insufficient_context": <True|False>,
  "answer_reasoning": "<Если "insufficient_context" = True, то оставь данное поле пустым. Иначе запиши сюда свои КОРОТКИЕ рассуждения о том, как выстроить ответ>",
  "missing_information": [Если "insufficient_context" = True, то запиши сюда дополнительные вопросы, без которых невозможно дать ответ на вопрос. Иначе оставь пустым]
  "answer": "<Итоговый ответ пользователю>"
}}
```

## Примеры (Cтиль, которого надо придерживаться):
# плохой пример (не используем)
{{
  "context_reasoning": "...",
  "insufficient_context": False,
  "answer_reasoning": "...",
  "missing_information": [],
  "answer": "## Как изменился лимит социального вычета на спорт по сравнению с 2022 годом? С 2024 года лимит..."
}}

# плохой пример (не используем)
{{
  "context_reasoning": "...",
  "insufficient_context": False,
  "answer_reasoning": "...",
  "missing_information": [],
  "answer": "## Изменение лимита социального вычета на спорт по сравнению с 2022 годом. С 2024 года лимит..."
}}

# хороший пример
{{
  "context_reasoning": "...",
  "insufficient_context": False,
  "answer_reasoning": "...",
  "missing_information": [],
  "answer": "С 2024 года лимит увеличен с 120 000 ₽ до **150 000 ₽ в год**. Это даёт возможность вернуть до 19 500 ₽ (150 000 × 13%) вместо прежних 15 600 ₽ (120 000 × 13%)."
}}

## **ВНУТРЕННИЕ ЭТАПЫ ИЗВЛЕЧЕНИЯ ** *(не выводить пользователю)*

1. Отбрось ненужную информацию из документов контекста.  
2. Сравни термины, цифры и ключевые детали для обеспечения согласованности.  
3. Выясни, хватает ли информации, чтобы дать ответ на вопрос.
4. Сформулируй логичный ответ из извлеченных фактов.
5. Как финансовый эксперт, дай клиенту понятный ответ на вопрос. Не начинай свой ответ с повторения вопроса.

## **ПЕРЕД ВЫВОДОМ ПРОВЕРЬ ПРАВИЛЬНОСТЬ** 
- JSON синтаксически корректен.  
- Никаких выдуманных деталей.  
- Ответ является лаконичным и понятным.

"""

    SUMMARY_PROMPT = """
Задача: Создать сфокусированное резюме документа в соответствии с запросом пользователя.

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_query}

ДОКУМЕНТ:
{document}

---

ЭТАП 1: ОПРЕДЕЛЕНИЕ НЕОБХОДИМОСТИ ИНФОРМАЦИИ
Прежде всего, определи, нужна ли тебе информация из документа для ответа на запрос. 
Если нет, то верни necessity: "False" и на следующие вопросы возвращай null
Если да, то верни necessity: "True"

---

ЭТАП 2: СОСТАВЛЕНИЕ РЕЗЮМЕ
Инструкции:
- Найди все части документа, которые имеют отношение к запросу
- Игнорируй информацию, которая не связана с вопросом
- Сохрани все важные детали, цифры, факты
- Перед тем, как дать ответ хорошо обумай все детали.


"""

    defaul_answer = 'К сожалению, у меня нет достаточной информации для ответа на ваш вопрос. Рекомендую обратиться в службу поддержки банка для получения детальной консультации.'

    def __init__(self):
        rag_prompt_template = PromptTemplate(
            template=FinancialRAG.SYSTEM_PROMPT,
            input_variables=["context_str", "question"],
        )

        self.answer_llm = ChatOpenAI(
            model=LLM_MODEL,
            # base_url="https://openrouter.ai/api/v1",
            base_url=BASE_URL,
            api_key=LLM_API_KEY,
            temperature=0,
        ).with_structured_output(DocumentAnswer)

        self.embedder_client = OpenAIEmbeddings(
            base_url=BASE_URL,
            model=EMBEDDER_MODEL,
            api_key=EMBEDDER_API_KEY,
        )

        self.summary_llm = ChatOpenAI(
            model=LLM_MODEL,
            # base_url="https://openrouter.ai/api/v1",
            base_url=BASE_URL,
            api_key=LLM_API_KEY,
        ).with_structured_output(DocumentSummary)

        self.chain = rag_prompt_template | self.answer_llm

        self.vector_store: VectorStore | None = None
        self.documents = []
        self.metadata = []
        
    def load_and_process_data(self, train_data_path: str):
        """Загрузка и обработка тренировочных данных"""
        print("Загрузка данных...")
        train_df = pd.read_csv(train_data_path)
        
        # Объединяем аннотацию, теги и текст для лучшего контекста
        processed_docs = []
        metadata_list = []
        
        for idx, row in train_df.iterrows():
            # Создаем обогащенный документ
            enriched_text = row['text']
            # enriched_text = f"""
            # Аннотация: {row['annotation']}
            # Теги: {row['tags']}
            # Содержание: {row['text']}
            # """
            
            processed_docs.append(enriched_text)
            metadata_list.append({
                'id': row['id'],
                'annotation': row['annotation'],
                'tags': row['tags'],
                'original_text': row['text']
            })
        
        return processed_docs, metadata_list
    
    def setup_vector_store(self, documents: List[str], metadata: List[Dict], save_path: str = "faiss_index"):
        """Создание или загрузка существующего векторного хранилища"""
        if os.path.exists(save_path):
            print("Загрузка существующего FAISS индекса...")
            self.vector_store = FAISS.load_local(save_path, self.embedder_client, allow_dangerous_deserialization=True)
            return

        print("Создание нового FAISS индекса...")
        embedding_dim = len(self.embedder_client.embed_query("hello world"))
        self.vector_store = FAISS(self.embedder_client,
                                  index=faiss.IndexFlatL2(embedding_dim),
                                  docstore=InMemoryDocstore(),
                                  index_to_docstore_id={},
                                  normalize_L2=True)

        # Разбиваем текст на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60,
            length_function=len,
            separators=[
                "\n## ",
                "\n### ",
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )
        
        # headers_to_split_on = [
            # ("##", "section"),
            # ("###", "subsection"),
        # ]

        # md_text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        all_chunks = []
        all_metadata = []
        all_documents = []

        for doc, meta in zip(documents, metadata):
            chunks = text_splitter.split_text(doc)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append(meta)
                all_documents.append(Document(chunk, metadata=meta))

        # Создаем эмбеддинги
        print("Создание эмбеддингов...")
        self.add_documents(all_documents)

        print(f"Создан индекс с {len(all_documents)} чанками")

        # Сохранение vector store на диск
        os.makedirs(save_path, exist_ok=True)
        self.vector_store.save_local(save_path)
        print(f"Vector store успешно сохранён в каталоге: {save_path}")

    def add_documents(self, docs: list[Document]):
        """Добавляет документы в хранилище, учитывая ограничения эмбеддера в 300_000 токенов за раз"""
        batch, tokens = [], 0

        for doc in tqdm(docs):
            doc_tokens = count_tokens(doc.page_content)
            if tokens + doc_tokens > MAX_EMBEDDING_TOKENS and batch:
                self.vector_store.add_documents(batch)
                batch, tokens = [doc], doc_tokens
            else:
                batch.append(doc)
                tokens += doc_tokens

        if batch:
            self.vector_store.add_documents(batch)

    async def get_summary(self, query: str, text: str) -> Optional[DocumentSummary]:
        """Асинхронно получает суммаризацию документа с использованием Structured Output"""
        try:
    
            summary_prompt = PromptTemplate(
                template=FinancialRAG.SUMMARY_PROMPT,
                input_variables=["user_query", "document"]
            )
            
            chain = summary_prompt | self.summary_llm
            result = await chain.ainvoke({
                "user_query": query, 
                "document": text
            })
            
            return result
        except Exception as e:
            print(f"Ошибка при суммаризации документа: {e}")
            return None

    async def summarize_documents_async(self, query: str, documents_with_scores: List[Tuple[str, float]]) -> List[Tuple[Document, float]]:
        """Асинхронная суммаризация всех документов"""
        tasks = []
        
        # Создаем задачи для асинхронной обработки
        for text, score in documents_with_scores:
            task = self.get_summary(query, text)
            tasks.append((task, text, score))
        
        # Запускаем все задачи параллельно
        results = []
        for task, text, score in tasks:
            summary = await task
            if summary and summary.necessity:
                # Создаем новый документ с суммаризацией
                summary_doc = Document(
                    page_content=summary.final_summary,
                )
                results.append((summary_doc, score))
        
        # Сортируем по релевантности (меньший score = лучше)
        results.sort(key=lambda x: x[1])
        return results


    def search_documents(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        """Поиск релевантных документов"""
        if self.vector_store is None:
            raise ValueError("Векторное хранилище не инициализировано")
        
        chunks = self.vector_store.similarity_search_with_score(query, k=k)

        # unique_docs = {}
        # for doc, score in chunks:
        #     doc_id = doc.metadata['id']
        #     # Сохраняем документ с лучшим score (наименьшее расстояние)
        #     if doc_id not in unique_docs or score < unique_docs[doc_id][1]:
        #         unique_docs[doc_id] = (doc.metadata['original_text'], score)
        
        # # Берем топ-K документов
        # top_docs = sorted(unique_docs.values(), key=lambda x: x[1])[:k]
        
        # results = asyncio.run(self.summarize_documents_async(query, top_docs))
        
        return chunks


    def generate(self, question: str, context: list[Document]) -> dict:
        if not context:
            print('Нет информации по вопросу')
            return {'context_reasoning': '', 'insufficient_context': '', 'context': [], 'missing_information': [],  'answer_reasoning': '', 'answer': self.defaul_answer}
        context_str = "\n\n".join([
            f"Документ {i + 1} :\n{doc.page_content}"
            for i, (doc, score) in enumerate(context)
        ])
        result = None
        try:
            result = self.chain.invoke({"context_str": context_str, "question": question})
        except Exception as e:
            import traceback
            print("ERROR: chain.invoke raised exception:")
            traceback.print_exc()
            # Попробуем получить raw response, если есть
            resp = getattr(e, "response", None) or getattr(e, "raw_response", None)
            if resp is not None:
                try:
                    print("ERROR: raw response:", resp.text)
                except Exception:
                    try:
                        print("ERROR: raw response (repr):", repr(resp)[:2000])
                    except Exception:
                        pass
            print(result)
        
        if not result:
            return {'context_reasoning': '', 'insufficient_context': '', 'context': [], 'missing_information': [], 'answer_reasoning': '', 'answer': self.defaul_answer}

        if result.insufficient_context:
            final_answer = self.defaul_answer
        else:
            final_answer = result.answer

        return {
                'context_reasoning': result.context_reasoning, 
                'insufficient_context': result.insufficient_context, 
                'missing_information': result.missing_information, 
                'answer_reasoning': result.answer_reasoning, 
                'answer': final_answer, 
                'context': context
            }

    def create_graph(self, k: int = 10) -> CompiledStateGraph[State]:
        def retrieve(state: State):
            return {"context": self.search_documents(state["question"], k)}

        def generate(state: State):
            return self.generate(state["question"], state["context"])

        # Compile application and test
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()


if __name__ == "__main__":    
    train_df, questions_df = pd.read_csv('train_data.csv'), pd.read_csv('questions.csv')

    # Инициализация RAG системы
    rag_system = FinancialRAG()

    # Создание векторного хранилища (только при первом запуске)
    documents, metadata = rag_system.load_and_process_data('./train_data.csv')

    if EMBEDDER_MODEL == 'text-embedding-ada-002':
        folder = 'faiss_financial_index_ada'
    else:
        folder = 'faiss_financial_index_small'

    rag_system.setup_vector_store(documents, metadata, folder)

    # Обработка вопросов
    questions_list = questions_df['Вопрос'].tolist()
    answer_list = []
    chunks_list = []
    scores_list = []
    insufficient_contexts = []
    context_reasonings = []
    answer_reasonings = []
    missing_informations = []

    rag_graph = rag_system.create_graph()

    print("\n=== ГЕНЕРАЦИЯ ОТВЕТОВ ===")
    batch_size = 10
    total_batches = (len(questions_list) + batch_size - 1) // batch_size

    # Создаем прогресс-бар с отображением количества пакетов
    with tqdm(total=total_batches, desc="Обработка вопросов") as pbar:
        for i in range(0, len(questions_list), batch_size):
            # Обрабатываем текущий пакет
            # noinspection PyTypeChecker
            results = rag_graph.batch(
                [{"question": current_question} for current_question in questions_list[i:i + batch_size]])
            answer_list.extend([result['answer'] for result in results])
            insufficient_contexts.extend([result['insufficient_context'] for result in results])
            context_reasonings.extend([result['context_reasoning'] for result in results])
            answer_reasonings.extend([result['answer_reasoning'] for result in results])
            missing_informations.extend([result['missing_information'] for result in results])


            for result in results:
                context = []
                scores = []
                for chunk in result['context']:
                    # e = chunk[0]
                    context.append(chunk[0].page_content)
                    scores.append(float(chunk[1]))
                chunks_list.append(context)
                scores_list.append(scores)
            # Обновляем прогресс-бар
            pbar.update(1)

    # Сохранение результатов
    questions_df['answer'] = answer_list
    # questions['Ответы на вопрос'] = answer_list
    # Для сбора статистики и расследования инцидентов
    questions_df['chunks'] = chunks_list
    questions_df['scores'] = scores_list
    questions_df['insufficient_context'] = insufficient_contexts
    questions_df['context_reasoning'] = context_reasonings
    questions_df['answer_reasoning'] = answer_reasonings
    questions_df['missing_information'] = missing_informations

    model_name = LLM_MODEL.split('/')[-1].split('-')[0]
    embedder_name = EMBEDDER_MODEL.split('-')[2]

    questions_df.to_csv(f'sub_{model_name}_{embedder_name}_prompt3.csv', index=False)
    # questions_df.to_csv(f'submission.csv', index=False)