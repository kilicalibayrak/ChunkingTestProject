from typing import List
import re

try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
except Exception:
    SentenceTransformer = None


def _sent_tokenize(text: str) -> List[str]:
    """Cümlelere böl (önce nltk, yoksa regex)."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text)]
    except Exception:
        # basit regex fallback (TR/EN için iş görür)
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-ZİÖÜÇĞŞ])', text) if s.strip()]


def run(
    text: str,
    target_chars: int = 900,
    sim_th: float = 0.25,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[str]:
    """
    Semantic chunking:
      - Metni cümlelere ayırır.
      - cümle embedding'leri ile ilerler.
      - Mevcut chunk'ın centroid'ine benzerlik >= sim_th ise aynı chunk'a ekler,
        değilse yeni chunk başlatır.
      - Chunk uzunluğu target_chars'i aşarsa yeni chunk'a geçer.
    """
    if SentenceTransformer is None:
        raise RuntimeError(
            "semantic için 'sentence-transformers' gerekli. Kur: pip install sentence-transformers"
        )

    sents = _sent_tokenize(text)
    sents = [s for s in sents if s]
    if not sents:
        return []

    import numpy as np

    model = SentenceTransformer(model_name)
    embs = model.encode(sents, convert_to_numpy=True, normalize_embeddings=True)

    chunks: List[str] = []
    cur_idx: List[int] = []
    cur_len = 0
    centroid = None

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float((a * b).sum())

    for i, (sent, emb) in enumerate(zip(sents, embs)):
        if not cur_idx:
            # ilk cümle ile başla
            cur_idx = [i]
            centroid = emb.copy()
            cur_len = len(sent)
            continue

        sim = cos(emb, centroid)
        # hem uzunluk sınırı hem de anlamsal eşik
        fits_len = (cur_len + len(sent) + 1) <= target_chars
        fits_sem = sim >= sim_th

        if fits_len and fits_sem:
            # aynı chunk'ta devam
            cur_idx.append(i)
            # centroid güncelle
            centroid = (centroid * (len(cur_idx) - 1) + emb) / len(cur_idx)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            cur_len += len(sent) + 1
        else:
            # chunk'ı bitir, yenisine başla
            chunks.append(" ".join(sents[j] for j in cur_idx))
            cur_idx = [i]
            centroid = emb.copy()
            cur_len = len(sent)

    if cur_idx:
        chunks.append(" ".join(sents[j] for j in cur_idx))

    return chunks