"""
Module: MultiTagHTMLReader.py
Description:
Description:
    This module allows you to parse with several HTML tags from HTML files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


class MultiTagHTMLReader(BaseReader):
    """
    Custom LlamaIndex reader that parses HTML files with BeautifulSoup4,
    strips boilerplate/scripts, and extracts content from specific tags.
    """

    def __init__(
            self,
            target_tags: Optional[List[str]] = None,
            parser: str = "html.parser",
            clean_tags: Optional[List[str]] = None,
    ):
        """
        :param target_tags: HTML tags to extract (e.g. ['h1', 'h2', 'p', 'li']).
                            If None, extracts all visible text.
        :param parser: BS4 parser back-end ('html.parser', 'lxml', 'html5lib').
        :param clean_tags: Unwanted tags to strip before extraction.
        """
        self.target_tags = target_tags or ["h1", "h2", "h3", "p", "li", "article"]
        self.parser = parser
        self.clean_tags = clean_tags or ["script", "style", "nav", "footer", "header", "aside"]

    def load_data(
            self, file: Path, extra_info: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Reads, cleans, and extracts target HTML elements into LlamaIndex Documents."""
        with open(file, "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, self.parser)

        # 1. Strip unwanted boilerplate elements
        for tag in soup(self.clean_tags):
            tag.decompose()

        # 2. Extract multiple tags at once using a list or regex match
        extracted_elements = soup.find_all(self.target_tags)

        # 3. Process and format the extracted text blocks
        extracted_text = "\n\n".join(
            el.get_text(strip=True)
            for el in extracted_elements
            if el.get_text(strip=True)
        )

        # 4. Construct metadata
        metadata = {
            "file_name": file.name,
            "file_path": str(file.resolve()),
            "tags_extracted": ", ".join(self.target_tags),
        }
        if extra_info:
            metadata.update(extra_info)

        return [Document(text=extracted_text, metadata=metadata)]
