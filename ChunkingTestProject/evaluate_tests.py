# -*- coding: utf-8 -*-
"""
tests/ klasöründeki chunk çıktılarını (TXT) okuyup puanlar.
Her dosya için metrikler:
  - num_chunks, avg_chars, std_chars, min_chars, max_chars
  - cohesion (yüksek iyi): chunk içi cümle benzerliği
  - boundary_sharpness (yüksek iyi): komşu chunk'lar arası ayrışma = 1 - cosine
  - redundancy (düşük iyi): tüm chunklar arası ortalama benzerlik

Çıktılar:
  - out/report_metrics.json   (ham metrikler)
  - out/report_summary.md     (özet tablo + sıralama)
"""

from pathlib import Path
import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

TESTS_DIR = Path("tests")
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True, parents=True)

CHUNK_SEP = re.compile(r"^=+\s*CHUNK\s+\d+\s*=+$", re.IGNORECASE | re.MULTILINE)

# --- yardımcılar --------------------------------------------------------------

def read_chunks_from_txt(path: Path):
    """tests/foo.txt içindeki chunk'ları ayıkla."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # "===== CHUNK n =====" ayırıcılarına göre böl
    parts = CHUNK_SEP.split(raw)
    # bazı dosyalarda başta boş parça olabilir
    chunks = [p.strip() for p in parts if p.strip()]
    return chunks

def sent_split(text: str):
    """Basit cümle bölücü (nltk yoksa iş görür)."""
    sents = re.split(r'(?<=[.!?])\s+(?=[A-ZİÖÜÇĞŞ])', text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents if sents else [text.strip()]

def tfidf_embeddings(texts):
    """TF-IDF vektörleri (L2 normalize)."""
    vec = TfidfVectorizer().fit_transform(texts)  # (n, d)
    X = vec.toarray().astype("float32")
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / nrm

def stats_lengths(chunks):
    if not chunks:
        return dict(num=0, avg=0.0, std=0.0, min=0, max=0)
    lens = np.array([len(c) for c in chunks], dtype=np.int32)
    return dict(
        num=int(len(chunks)),
        avg=float(lens.mean()),
        std=float(lens.std()),
        min=int(lens.min()),
        max=int(lens.max()),
    )

def cohesion_score(chunks):
    """Her chunk içindeki cümlelerin centroid'e cosine benzerliği ortalaması."""
    if not chunks:
        return 0.0
    scores = []
    for ch in chunks:
        sents = sent_split(ch)
        if len(sents) == 1:
            scores.append(1.0)  # tek cümle -> tam uyum varsay
            continue
        E = tfidf_embeddings(sents)  # (m,d)
        centroid = E.mean(0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = E @ centroid
        scores.append(float(np.mean(sims)))
    return float(np.mean(scores)) if scores else 0.0

def boundary_sharpness_score(chunks):
    """Komşu chunk'lar arası 1 - cosine ortalaması (yüksekse sınırlar 'keskin')."""
    if len(chunks) < 2:
        return 0.0
    E = tfidf_embeddings(chunks)  # (n,d)
    sims = np.sum(E[:-1] * E[1:], axis=1)  # komşu cosine
    return float(np.mean(1.0 - sims))

def redundancy_score(chunks):
    """Tüm chunk'lar arası ortalama cosine (düşük daha iyi)."""
    n = len(chunks)
    if n < 2:
        return 0.0
    E = tfidf_embeddings(chunks)  # (n,d)
    S = E @ E.T                   # (n,n)
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(S[mask]))

def score_aggregate(metrics, target_chars=900.0):
    """
    Tek sayı skor (yukarı iyidir):
      0.45*cohesion + 0.30*boundary - 0.15*redundancy + 0.10*size_bonus
    """
    s = metrics["stats"]
    cohesion = metrics["cohesion"]
    boundary = metrics["boundary_sharpness"]
    redundancy = metrics["redundancy"]
    size_bonus = -abs(s["avg"] - target_chars) / target_chars
    return 0.45*cohesion + 0.30*boundary - 0.15*redundancy + 0.10*size_bonus

# --- ana akış ----------------------------------------------------------------

def main():
    results = {}
    txt_files = sorted(TESTS_DIR.glob("*.txt"))
    if not txt_files:
        print("tests/ klasöründe .txt bulunamadı.")
        return

    for f in txt_files:
        try:
            chunks = read_chunks_from_txt(f)
            m = {
                "stats": stats_lengths(chunks),
                "cohesion": cohesion_score(chunks),
                "boundary_sharpness": boundary_sharpness_score(chunks),
                "redundancy": redundancy_score(chunks),
            }
            m["score"] = score_aggregate(m)
            results[f.name] = m
        except Exception as e:
            results[f.name] = {"error": str(e)}

    # JSON kaydet
    (OUT_DIR / "report_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Markdown özet + sıralama
    ok_items = [(k, v) for k, v in results.items() if "error" not in v]
    ok_items.sort(key=lambda kv: kv[1]["score"], reverse=True)

    lines = ["# Chunking Değerlendirme Özeti\n"]
    lines.append("| Dosya | Skor | #Chunks | AvgLen | Cohesion | Boundary | Redundancy |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, m in ok_items:
        s = m["stats"]
        lines.append(
            f"| {name} | {m['score']:.4f} | {s['num']} | {s['avg']:.0f} | "
            f"{m['cohesion']:.3f} | {m['boundary_sharpness']:.3f} | {m['redundancy']:.3f} |"
        )
    bad = [(k, v) for k, v in results.items() if "error" in v]
    if bad:
        lines.append("\n## Hata Verenler")
        for name, m in bad:
            lines.append(f"- **{name}** → {m['error']}")

    (OUT_DIR / "report_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("✓ Bitti:")
    print(f"  - out/report_metrics.json")
    print(f"  - out/report_summary.md")

if __name__ == "__main__":
    main()