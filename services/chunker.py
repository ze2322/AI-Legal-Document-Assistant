from langchain_text_splitters import RecursiveCharacterTextSplitter
import utils.config as config
class TextChunker:
    """
    Splits document text into overlapping chunks.
    """

    def __init__(
        self,
        chunk_size: int = 700,
        chunk_overlap: int = 100
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def split_text(self, text: str) -> list[str]:
        """
        Split text into chunks.

        Args:
            text: Extracted document text.

        Returns:
            List of text chunks.
        """
        return self.splitter.split_text(text)