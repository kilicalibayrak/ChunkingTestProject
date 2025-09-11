from typing import List
from .agentic import _is_heading
from .sentence_based import _sent_tokenize as _sents

def _trim_to_max(s: str, max_chars: int) -> str:
    if max_chars is None or len(s) <= max_chars:
        return s
    cut = max_chars
    if s[cut-1:cut].isalnum() and s[cut:cut+1].isalnum():
        back = s[max(0, cut-40):cut]
        sp = back.rfind(" ")
        if sp != -1:
            cut = max(0, cut-40) + sp
    return s[:cut].rstrip()

def run(
    text: str,
    target_chars: int = 900,
    window: int = 6,
    stride: int = 3,
) -> List[str]:
    """
    Hybrid Chunking:
      1) Başlıklara göre bölümlere ayır.
      2) Her bölümde cümle bazlı sliding window uygula (window/stride).
      3) target_chars aşılırsa yumuşak kes.

    Parametreler:
      - target_chars: chunk uzunluğu üst sınırı
      - window: pencere cümle sayısı
      - stride: bir sonraki pencereye geçişte kaydırma cümle sayısı
    """
    norm = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in norm.split("\n") if ln.strip()]

    # 1) Bölümlere ayır (heading sezgisi)
    sections: List[str] = []
    cur: List[str] = []
    for ln in lines:
        if _is_heading(ln):
            if cur:
                sections.append("\n".join(cur))
                cur = []
        cur.append(ln)
    if cur:
        sections.append("\n".join(cur))
    if not sections:
        sections = [text]

    # 2) Bölüm içinde sliding window
    out: List[str] = []
    for sec in sections:
        sents = [s.strip() for s in _sents(sec) if s and s.strip()]
        i, n = 0, len(sents)
        while i < n:
            win = sents[i:i+window]
            if not win:
                break
            joined = " ".join(win).strip()
            joined = _trim_to_max(joined, target_chars)
            if joined:
                out.append(joined)
            if i + window >= n:
                break
            i += stride

    return out