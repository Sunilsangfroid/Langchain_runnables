from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


# Intialize the llm
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Define the prompt
prompt = PromptTemplate(
    input_variables=["topic"],
    template = "Suggest a catchy title for a blog post about {topic}."
)

# Define the input
topic = input("Enter the topic: ")

# Format the prompt manually using PromptTemplate
formatted_prompt = prompt.format(topic=topic)

# Call the llm directly
blog_title = llm.predict(formatted_prompt)

# Print the Output
print(blog_title)