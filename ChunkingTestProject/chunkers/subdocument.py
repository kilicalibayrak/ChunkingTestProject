from typing import List
from .sentence_based import run as sentence_chunk
from .agentic import _is_heading  # başlık sezgimizi yeniden kullanıyoruz

def run(
    text: str,
    target_chars: int = 900,
    overlap_sent: int = 0,
) -> List[str]:
    """
    Subdocument Chunking:
      1) Metni başlıklara göre alt-dokümanlara böl.
      2) Her alt-dokümanı bağımsız bir belge gibi ele alıp sentence-based chunkla.
      3) Başlık hiç yoksa tüm metni sentence-based chunkla (fallback).
    """
    norm = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in norm.split("\n") if ln.strip()]

    # Alt-dokümanları ayır: heading gördükçe yeni subdoc başlat
    subdocs: List[List[str]] = []
    cur: List[str] = []
    for ln in lines:
        if _is_heading(ln):
            if cur:
                subdocs.append(cur)
                cur = []
        cur.append(ln)
    if cur:
        subdocs.append(cur)

    # Başlık yoksa fallback
    if not subdocs:
        return sentence_chunk(text, target_chars=target_chars, overlap_sent=overlap_sent)

    # Her alt-dokümanı bağımsız işle
    chunks: List[str] = []
    for sd in subdocs:
        sd_text = "\n".join(sd).strip()  # başlığı da içinde tut
        # alt-doküman içinde overlap'ı düşük tutmak genelde iyi (0/1)
        sd_chunks = sentence_chunk(sd_text, target_chars=target_chars, overlap_sent=overlap_sent)
        chunks.extend(sd_chunks)

    return chunks