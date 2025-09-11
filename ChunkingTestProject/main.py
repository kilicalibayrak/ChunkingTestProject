from pathlib import Path
import importlib

# hangi chunker dosyalarını çalıştıracağımız
METHODS = [
    "fixed_length",
    "sentence_based",
    "paragraph_based",
    "sliding_window",
    "semantic",
    "recursive",
    "context_enriched",
    "agentic",
    "subdocument",
    "hybrid",
]

def main():
    # text oku
    text = Path("data/rag_dataset.json").read_text(encoding="utf-8")

    tests_dir = Path("tests")
    tests_dir.mkdir(exist_ok=True)

    for method in METHODS:
        try:
            mod = importlib.import_module(f"chunkers.{method}")
            chunks = mod.run(text)
            out_file = tests_dir / f"{method}.txt"
            with out_file.open("w", encoding="utf-8") as f:
                for i, ch in enumerate(chunks, 1):
                    f.write(f"===== CHUNK {i} =====\n{ch}\n\n")
            print(f"{method} → {len(chunks)} chunks kaydedildi: {out_file}")
        except Exception as e:
            print(f"{method} hata verdi: {e}")

if __name__ == "__main__":
    main()