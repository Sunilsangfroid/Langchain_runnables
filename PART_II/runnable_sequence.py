from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', api_key=google_api_key)

prompt1 = PromptTemplate(
    template='Write a joke about {topic}.',
    input_variables=['topic'],
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the given joke: {joke}",
    input_variables=['joke'],
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({'topic':'cricket'})

print(result)