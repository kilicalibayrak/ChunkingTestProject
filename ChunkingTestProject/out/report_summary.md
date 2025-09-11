# Chunking Değerlendirme Özeti

| Dosya | Skor | #Chunks | AvgLen | Cohesion | Boundary | Redundancy |
|---|---:|---:|---:|---:|---:|---:|
| agentic.txt | 0.5605 | 1523 | 690 | 0.786 | 0.774 | 0.015 |
| subdocument.txt | 0.5605 | 1523 | 690 | 0.786 | 0.774 | 0.015 |
| recursive.txt | 0.5467 | 1464 | 869 | 0.768 | 0.691 | 0.017 |
| sentence_based.txt | 0.5285 | 1761 | 724 | 0.768 | 0.683 | 0.016 |
| paragraph_based.txt | 0.5153 | 1245 | 899 | 0.663 | 0.733 | 0.018 |
| fixed_length.txt | 0.4870 | 1607 | 796 | 0.684 | 0.645 | 0.017 |
| sliding_window.txt | 0.4175 | 966 | 847 | 0.535 | 0.618 | 0.019 |
| hybrid.txt | 0.3915 | 1288 | 784 | 0.548 | 0.535 | 0.018 |
| context_enriched.txt | 0.3691 | 1761 | 1370 | 0.661 | 0.422 | 0.019 |

## Hata Verenler
- **semantic.txt** → empty vocabulary; perhaps the documents only contain stop words