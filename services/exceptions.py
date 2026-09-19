"""Exception types raised by the services layer.

Having a single base class lets the Streamlit UI catch everything the pipeline
can legitimately fail with and show a readable message, instead of dumping a
traceback on the user.
"""


class DocumentAssistantError(Exception):
    """Base class for all expected application errors."""


class UnsupportedFileTypeError(DocumentAssistantError):
    """Raised when a file extension is not PDF or DOCX."""


class DocumentParsingError(DocumentAssistantError):
    """Raised when a document exists but cannot be read."""


class EmptyDocumentError(DocumentAssistantError):
    """Raised when a document yields no extractable text (e.g. a scanned PDF)."""


class IndexNotReadyError(DocumentAssistantError):
    """Raised when a search is attempted before a document has been indexed."""


class LLMUnavailableError(DocumentAssistantError):
    """Raised when the Ollama server cannot be reached or the model is missing."""
