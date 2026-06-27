import ollama
from utils.config import OLLAMA_MODEL


class LLMService:
    """
    Handles communication with the Ollama model.
    """

    def __init__(self):
        self.model = OLLAMA_MODEL

    def generate(self, context: str, question: str) -> str:
        """
        Generate an answer using the retrieved context.
        """

        prompt = f"""
You are an experienced legal AI assistant.

Your task is to answer questions ONLY using the provided document context.

Instructions:
- Provide a detailed and well-structured answer.
- Explain the information in your own words.
- Include important details from the context.
- If appropriate, summarize key points in bullet points.
- Do NOT invent information that is not present in the context.
- If the answer cannot be found, say:
"I couldn't find this information in the uploaded document."

Context:
-------------------------
{context}
-------------------------

Question:
{question}

Answer:
"""

        response = ollama.chat(
          model=self.model,
            messages=[
        {
            "role": "user",
            "content": prompt
        }
          ],
    options={
        "num_predict": 512,
        "temperature": 0.3,
    }
)

        return response["message"]["content"]