from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

prompt = PromptTemplate(
    input_variables=["topic"],
    template = "Suggest a catchy title for a blog post about {topic}."
)

chain = LLMChain(llm=llm, prompt=prompt)

topic = input("Enter the topic: ")
output = chain.run(topic=topic)

print("Generated Blog Title: ", output)
