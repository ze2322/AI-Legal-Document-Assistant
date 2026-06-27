from services.parser import DocumentParser

parser = DocumentParser()
text = parser.parse("D:\AOU\projects\PDF Question Answering Assistant Task\data\documents\Policy Guidelines for Agreements with Implementing Partners.pdf")

print(text)

