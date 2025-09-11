from typing import List
import re
from .sentence_based import run as sentence_chunk

def _paragraphs(text: str) -> List[str]:
    """Metni paragraflara ayırır (boş satıra göre)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def run(text: str, target_chars: int = 900) -> List[str]:
    paras = _paragraphs(text)
    if not paras:
        # paragraf yoksa fallback
        return sentence_chunk(text, target_chars=target_chars, overlap_sent=0)

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for p in paras:
        if len(p) > target_chars:
            # çok uzun paragraf → parçalara böl
            i = 0
            while i < len(p):
                end = min(i + target_chars, len(p))
                part = p[i:end].strip()
                if cur:
                    chunks.append("\n\n".join(cur))
                    cur = []
                    cur_len = 0
                if part:
                    chunks.append(part)
                i = end
            continue

        if cur_len + len(p) + 2 <= target_chars:
            cur.append(p)
            cur_len += len(p) + 2
        else:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = len(p) + 2

    if cur:
        chunks.append("\n\n".join(cur))

    return chunks