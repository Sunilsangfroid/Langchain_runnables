from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', api_key=google_api_key)

passthrough = RunnablePassthrough()

# print(passthrough.invoke({'topic':'AI'}))

prompt1 = PromptTemplate(
    template='Write a joke about {topic}.',
    input_variables=['topic'],
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the given joke: {joke}",
    input_variables=['joke'],
)

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'cricket'})

print(result)