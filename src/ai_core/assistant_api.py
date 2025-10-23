"""
assistant_api.py
-----------------------------------
Provides an AI assistant endpoint for teaching, consulting, or analytics support.
"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

class AssistantAPI:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY"))

    def respond(self, user_input: str):
        """Generate assistant response"""
        prompt = ChatPromptTemplate.from_template(
            "You are an expert in humanitarian supply chains and AI analytics. "
            "Answer clearly and professionally.\n\nUser: {question}\nAssistant:"
        )
        chain = prompt | self.llm
        response = chain.invoke({"question": user_input})
        return response.content


if __name__ == "__main__":
    bot = AssistantAPI()
    print(bot.respond("Explain how AI can optimize medical supply forecasting."))
