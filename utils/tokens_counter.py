import re

def get_token_counter():
    """
    Returns a token counting function. Prefers tiktoken if available,
    otherwise falls back to a simple whitespace-based heuristic.
    """
    try:
        import tiktoken
        # Using the encoding for gpt-3.5-turbo and gpt-4
        ENCODING = tiktoken.get_encoding("cl100k_base")
        def token_counter(text: str) -> int:
            return len(ENCODING.encode(text))
        print("Using tiktoken (cl100k_base) for token counting.")
        return token_counter
    except ImportError:
        def token_counter(text: str) -> int:
            if not text:
                return 0
            # A simple heuristic: count sequences of non-whitespace characters
            return len(re.findall(r"\S+", text))
        print("tiktoken not found. Using a whitespace-based heuristic for token counting.")
        return token_counter