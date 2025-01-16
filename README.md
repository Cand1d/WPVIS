# Wahlprogramm-Analyse 2025

Dieses Repository demonstriert eine automatisierte Analyse von deutschen Wahlprogrammen zur Bundestagswahl 2025. Ziel ist es, aus PDF-Dateien die Texte zu extrahieren, in Chunks aufzuteilen, mithilfe von Embeddings und thematischen Zuordnungen zu verarbeiten und schließlich interaktiv zu visualisieren.

## Übersicht

1. **PDF-Extraktion**  
   Lädt Wahlprogramme als PDF ein und extrahiert den Text (z. B. via `pdfplumber`).

2. **Chunking**  
   Teilt die extrahierten Texte in überlappende Abschnitte (Chunks), um thematisch feinere Einheiten zu erhalten.

3. **Embeddings**  
   Verwendet Voyage (oder ein anderes Embedding-Modell) zur semantischen Repräsentation der Chunks.

4. **Themen-Zuordnung**  
   Orientiert sich an einer vordefinierten Liste von Themen (z. B. „Außen- und Sicherheitspolitik", „Wirtschaftspolitik", „Bildung"). Jeder Chunk wird durch Cosine Similarity dem passendsten Thema zugeordnet.

5. **Interaktive Visualisierung**  
   - Führt beispielsweise eine PCA durch, um die Embeddings auf 2D zu reduzieren.  
   - Nutzt Plotly-`updatemenus`, um Dropdowns für Themen und Jahre anzubieten.  
   - Optional: Hover-Infos mit Kurz-Zusammenfassungen und Parteiinformationen.
   - Die finale Visualisierung befindet sich in index.html

## Installation

- **Python-Version**: Ab 3.8 oder höher empfohlen.
- **Abhängigkeiten**:
  ```bash
  pip install pdfplumber voyageai plotly scikit-learn numpy pandas
  # Ergänze ggf. weitere Pakete wie dash, openai usw.
  ```

## Projektstruktur

```
.
├── README.md
├── data/
│   ├── SPD_WP_2025.pdf
│   ├── CDU_WP_2025.pdf
│   └── ...
├── scripts/
│   ├── extract_text.py         # PDF-Extraktion
│   ├── chunk_text.py           # Funktion, um Text zu zerteilen
│   ├── compute_embeddings.py   # Erzeugung und Speicherung von Embeddings
│   ├── analyze_topics.py       # Themen-Zuordnung & Aggregation
│   └── create_plots.py         # Plotly-Visualisierung (PCA etc.)
└── main.py                     # Hauptskript
```

## Quickstart

### 1. Text extrahieren

```python
# extract_text.py
import pdfplumber

files = ["SPD_WP_2025.pdf", "CDU_WP_2025.pdf", ...]
WP = {}
for file in files:
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        WP[file] = text

# WP kann anschließend gepickelt oder als JSON gespeichert werden.
```

### 2. Chunking

```python
# chunk_text.py
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
```

### 3. Embeddings

```python
# compute_embeddings.py
import voyageai
import numpy as np
import pickle

vo = voyageai.Client(api_key="DEIN_API_KEY")

all_chunks = [...]  # z. B. die Ausgabe aus chunk_text
chunk_embeddings = []

for chunk in all_chunks:
    result = vo.embed(chunk, model="voyage-3-large", input_type="document")
    emb = result.embeddings[0]
    chunk_embeddings.append(emb)

with open("chunk_embeddings.pkl", "wb") as f:
    pickle.dump(chunk_embeddings, f)
```

### 4. Themen-Zuordnung

Definiere eine Liste deiner Themen, erstelle pro Thema ein Embedding und ermittle via Cosine Similarity den besten Match für jeden Chunk.

Weist jedem Chunk ein Thema zu oder lässt ihn unklassifiziert, wenn die Similarity unter einem gewissen Schwellwert liegt.

### 5. Visualisierung mit Plotly

```python
# create_plots.py
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

with open("chunk_embeddings.pkl", "rb") as f:
    chunk_embeddings = pickle.load(f)

embedding_array = np.array(chunk_embeddings)  # shape: (N, dim)
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(embedding_array)

fig = go.Figure(
    data=go.Scatter(
        x=pca_coords[:,0],
        y=pca_coords[:,1],
        mode="markers",
        marker=dict(size=8, color="blue")
    )
)
fig.update_layout(title="PCA der Wahlprogramm-Chunks")
fig.show()
```

## Hinweise

- **Performance**: Das Generieren vieler Embeddings kann zeitintensiv sein. Zwischenergebnisse sollten in Pickle-Dateien o. ä. gesichert werden.
- **Genauigkeit**: Die Qualität der Themen-Zuordnung hängt stark von den Themenlabels (z. B. „Energiepolitik") oder ausführlicheren Beschreibungen ab.
- **Lizenz**: Bitte füge eine passende Lizenzdatei (z. B. MIT, Apache 2.0) hinzu, wenn du den Code öffentlich teilst.

## Kontakt

Für Fragen oder Diskussionen bitte ein Issue auf GitHub öffnen oder den Projektverantwortlichen anschreiben.

