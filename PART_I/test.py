import os
from langchain_core.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
# from langchain_core.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# from typing import Literal

from langchain.llms import ChatGoogleGenerativeAI


load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', api_key=google_api_key)