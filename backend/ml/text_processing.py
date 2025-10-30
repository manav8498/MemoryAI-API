"""
Text processing utilities for chunking and preprocessing.
"""
from typing import List, Dict, Any
import re
from dataclasses import dataclass

from backend.core.config import settings
from backend.core.logging_config import logger


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


class TextChunker:
    """
    Text chunking with various strategies.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategy: str = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.strategy = strategy or settings.CHUNKING_STRATEGY

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """
        Chunk text using the configured strategy.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of text chunks
        """
        metadata = metadata or {}

        if self.strategy == "fixed":
            return self._chunk_fixed(text, metadata)
        elif self.strategy == "recursive":
            return self._chunk_recursive(text, metadata)
        elif self.strategy == "semantic":
            return self._chunk_semantic(text, metadata)
        else:
            logger.warning(f"Unknown chunking strategy: {self.strategy}, using recursive")
            return self._chunk_recursive(text, metadata)

    def _chunk_fixed(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Simple fixed-size chunking with overlap.

        Args:
            text: Text to chunk
            metadata: Chunk metadata

        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(TextChunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will update after
                    start_char=start,
                    end_char=end,
                    metadata=metadata.copy(),
                ))
                chunk_index += 1

            start = end - self.chunk_overlap
            if start >= len(text) - self.chunk_overlap:
                break

        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    def _chunk_recursive(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Recursive chunking that respects natural boundaries.

        Tries to split on:
        1. Double newlines (paragraphs)
        2. Single newlines
        3. Sentences
        4. Words
        5. Characters (fallback)

        Args:
            text: Text to chunk
            metadata: Chunk metadata

        Returns:
            List of chunks
        """
        separators = [
            "\n\n",  # Paragraphs
            "\n",     # Lines
            ". ",     # Sentences
            "! ",     # Sentences
            "? ",     # Sentences
            " ",      # Words
            "",       # Characters
        ]

        return self._recursive_split(text, separators, metadata)

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        metadata: Dict[str, Any],
        start_char: int = 0,
    ) -> List[TextChunk]:
        """
        Recursively split text using hierarchical separators.

        Args:
            text: Text to split
            separators: List of separators to try
            metadata: Chunk metadata
            start_char: Starting character index

        Returns:
            List of chunks
        """
        chunks = []

        if len(text) <= self.chunk_size:
            # Text fits in one chunk
            chunks.append(TextChunk(
                content=text.strip(),
                chunk_index=0,
                total_chunks=1,
                start_char=start_char,
                end_char=start_char + len(text),
                metadata=metadata.copy(),
            ))
            return chunks

        # Try each separator
        for separator in separators:
            if separator == "":
                # Last resort: character-level split
                return self._chunk_fixed(text, metadata)

            if separator in text:
                # Split by this separator
                splits = text.split(separator)
                current_chunk = []
                current_length = 0
                current_start = start_char

                for i, split in enumerate(splits):
                    split_len = len(split) + len(separator)

                    if current_length + split_len > self.chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = separator.join(current_chunk)
                        chunks.append(TextChunk(
                            content=chunk_text.strip(),
                            chunk_index=len(chunks),
                            total_chunks=0,
                            start_char=current_start,
                            end_char=current_start + len(chunk_text),
                            metadata=metadata.copy(),
                        ))

                        # Start new chunk with overlap
                        overlap_items = []
                        overlap_length = 0
                        for item in reversed(current_chunk):
                            overlap_length += len(item) + len(separator)
                            if overlap_length > self.chunk_overlap:
                                break
                            overlap_items.insert(0, item)

                        current_chunk = overlap_items
                        current_length = sum(len(x) + len(separator) for x in overlap_items)
                        current_start = current_start + len(chunk_text) - current_length

                    current_chunk.append(split)
                    current_length += split_len

                # Add remaining chunk
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    chunks.append(TextChunk(
                        content=chunk_text.strip(),
                        chunk_index=len(chunks),
                        total_chunks=0,
                        start_char=current_start,
                        end_char=current_start + len(chunk_text),
                        metadata=metadata.copy(),
                    ))

                break

        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    def _chunk_semantic(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Semantic chunking based on sentence embeddings.

        Groups sentences with similar meanings together.
        (Simplified version - full implementation would use embeddings)

        Args:
            text: Text to chunk
            metadata: Chunk metadata

        Returns:
            List of chunks
        """
        # For now, fall back to sentence-based chunking
        # TODO: Implement full semantic chunking with embeddings

        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        start_idx = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(TextChunk(
                    content=chunk_text.strip(),
                    chunk_index=len(chunks),
                    total_chunks=0,
                    start_char=start_idx,
                    end_char=start_idx + len(chunk_text),
                    metadata=metadata.copy(),
                ))

                # Start new chunk
                current_chunk = []
                current_length = 0
                start_idx += len(chunk_text)

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(TextChunk(
                content=chunk_text.strip(),
                chunk_index=len(chunks),
                total_chunks=0,
                start_char=start_idx,
                end_char=start_idx + len(chunk_text),
                metadata=metadata.copy(),
            ))

        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting (could be improved with NLTK or spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def get_text_chunker(
    chunk_size: int = None,
    chunk_overlap: int = None,
    strategy: str = None,
) -> TextChunker:
    """
    Get text chunker instance.

    Args:
        chunk_size: Optional chunk size override
        chunk_overlap: Optional overlap override
        strategy: Optional strategy override

    Returns:
        TextChunker instance
    """
    return TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
    )
