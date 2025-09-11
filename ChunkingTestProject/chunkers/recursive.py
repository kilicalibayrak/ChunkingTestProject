from typing import List
from .sentence_based import run as sentence_chunk
from .fixed_length import run as fixed_chunk

def run(
    text: str,
    max_chars: int = 1200,   # bir chunk'ın üst sınırı
    min_chars: int = 400,    # ikinci pass için alt hedef
    overlap_sent: int = 1    # cümle bazlı pass'lerde overlap
) -> List[str]:
    """
    Recursive chunking:
      1) Büyük hedefle sentence-based chunkla (max_chars).
      2) max_chars'ı geçen her chunk'ı tekrar sentence-based ile orta hedefe böl (mid).
      3) Hâlâ uzun olan varsa fixed-length ile kesin (son çare).

    Not: Amaç önce anlamı korumak (cümle bazlı), en sonda zorunlu olursa karakter kesimi yapmak.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if min_chars <= 0 or min_chars > max_chars:
        raise ValueError("min_chars must be > 0 and <= max_chars")

    # 1) İlk pass: geniş hedefle cümle bazlı
    base_chunks = sentence_chunk(text, target_chars=max_chars, overlap_sent=overlap_sent)

    out: List[str] = []
    mid = (max_chars + min_chars) // 2  # ikinci pass için orta hedef

    for ch in base_chunks:
        if len(ch) <= max_chars:
            out.append(ch)
            continue

        # 2) İkinci pass: hâlâ uzunsa, daha küçük hedefle cümle bazlı
        sub_chunks = sentence_chunk(ch, target_chars=mid, overlap_sent=overlap_sent)
        for sub in sub_chunks:
            if len(sub) <= max_chars:
                out.append(sub)
            else:
                # 3) Son çare: fixed-length (overlap 0 ya da az olabilir)
                out.extend(
                    fixed_chunk(sub, chunk_chars=max_chars, overlap_chars=0)
                )

    return out