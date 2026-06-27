from services.llm import LLMService

llm = LLMService()

context = """
Implementing partners are organizations responsible
for delivering project outputs under Grant-out Agreements.
"""

question = "Who are implementing partners?"

answer = llm.generate(
    context=context,
    question=question
)

print(answer)