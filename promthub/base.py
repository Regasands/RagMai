from langchain.prompts import PromptTemplate


prompt_template = """Ты — полезный ассистент, который отвечает на вопросы на основе предоставленного контекста.
Используй только информацию из контекста. Если ответа нет в контексте, скажи "В предоставленных документах нет информации по этому вопросу".

Контекст: {context}

Вопрос: {question}

Ответ:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)