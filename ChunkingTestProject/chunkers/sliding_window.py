from typing import List
from .sentence_based import _sent_tokenize as _sents

def _trim_to_max(s: str, max_chars: int) -> str:
    """Metni max_chars sınırına yumuşak kes (kelime ortasını bozma)."""
    if max_chars is None or len(s) <= max_chars:
        return s
    cut = max_chars
    # kelime ortasında kesmeyi önle
    if s[cut-1:cut].isalnum() and s[cut:cut+1].isalnum():
        back = s[max(0, cut-40):cut]
        sp = back.rfind(" ")
        if sp != -1:
            cut = max(0, cut-40) + sp
    return s[:cut].rstrip()

def run(
    text: str,
    window: int = 8,
    stride: int = 4,
    max_chars: int = 900
) -> List[str]:
    """
    Cümle bazlı sliding window:
      - window: her chunk'taki cümle sayısı
      - stride: bir sonraki pencereye geçişte kaç cümle kaydırılacağı
      - max_chars: chunk uzunluğu üst sınırı (yumuşak kesilir)
    """
    sents = [s.strip() for s in _sents(text) if s and s.strip()]
    chunks: List[str] = []
    n = len(sents)
    i = 0

    while i < n:
        win = sents[i:i+window]
        if not win:
            break
        joined = " ".join(win).strip()
        joined = _trim_to_max(joined, max_chars)
        if joined:
            chunks.append(joined)
        # sona geldiysek çık
        if i + window >= n:
            break
        i += stride

    return chunks