from typing import List
from .sentence_based import run as sentence_chunk

def run(
    text: str,
    target_chars: int = 900,
    overlap_sent: int = 1,
    side_ctx: int = 1,
) -> List[str]:
    """
    Context-Enriched Chunking:
      1) Cümle bazlı chunkla (sentence_based).
      2) Her chunk'a bir önceki ve bir sonraki chunk'tan 'side_ctx' kadar cümle ekle.
         (bağlam kaybını azaltır; retrieval kalitesini iyileştirir)

    Parametreler:
      - target_chars: ana chunk hedef boyutu (sentence_based için)
      - overlap_sent: sentence_based aşamasındaki cümle overlap sayısı
      - side_ctx: her chunk'a sol/sağdan eklenecek cümle sayısı
    """
    base = sentence_chunk(text, target_chars=target_chars, overlap_sent=overlap_sent)
    if not base:
        return []

    # Basit cümle kesici (nokta-temelli) — sadece komşu chunk'lardan ctx seçimi için
    def _split_simple(x: str) -> List[str]:
        parts = [t.strip() for t in x.replace("\r\n", "\n").replace("\r", "\n").split(". ") if t.strip()]
        # Nokta kaybolmasın diye geri ekleme (opsiyonel, kısa tut)
        return [p if p.endswith(".") else (p + ".") for p in parts]

    enriched: List[str] = []
    n = len(base)

    for i, ch in enumerate(base):
        left_tail = []
        right_head = []

        if side_ctx > 0 and i > 0:
            left_sents = _split_simple(base[i - 1])
            left_tail = left_sents[-side_ctx:]

        if side_ctx > 0 and i + 1 < n:
            right_sents = _split_simple(base[i + 1])
            right_head = right_sents[:side_ctx]

        # birleştir
        pieces = []
        if left_tail:
            pieces.append(" ".join(left_tail))
        pieces.append(ch)
        if right_head:
            pieces.append(" ".join(right_head))

        enriched_text = " ".join(pieces).strip()
        enriched.append(enriched_text)

    return enriched