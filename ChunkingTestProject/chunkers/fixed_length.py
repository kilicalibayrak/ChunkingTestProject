from typing import List

def run(text: str, chunk_chars: int = 800, overlap_chars: int = 100) -> List[str]:
    src = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = []
    n = len(src)
    i = 0
    while i < n:
        end = min(i + chunk_chars, n)
        # kelime ortasında kesme
        if end < n and src[end-1].isalnum() and src[end:end+1].isalnum():
            back = src[max(i, end-25):end]
            sp = back.rfind(" ")
            if sp != -1:
                end = max(i, end-25) + sp
        chunk = src[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        i = max(end - overlap_chars, i + 1)
    return chunks

