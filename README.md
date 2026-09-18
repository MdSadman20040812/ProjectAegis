![Project Aegis](https://img.shields.io/badge/ProjectAegis-Cognitive%20Tutor-dc2626?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1c1c1c?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Latest-ff6f00?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-ff4b4b?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Cognitive tutor — deconstruct academic documents into first principles via ABCD framework.**

---

## 🏗️ Pipeline

```mermaid
graph LR
    subgraph Input
        DOC[Academic<br/>Document]
        Q[Student<br/>Query]
    end
    subgraph LangGraph Pipeline
        PARSE[Parse<br/>Structure]
        DECONSTRUCT[Deconstruct<br/>ABCD Framework]
        EXPLAIN[Explain<br/>First Principles]
        VERIFY[Verify<br/>Understanding]
    end
    subgraph Storage
        VDB[(ChromaDB<br/>Concept Vectors)]
    end
    subgraph Output
        EXPLANATION[First-Principles<br/>Explanation]
        QUIZ[Comprehension<br/>Check]
    end
    DOC --> PARSE
    PARSE --> DECONSTRUCT
    DECONSTRUCT --> EXPLAIN
    Q --> EXPLAIN
    EXPLAIN --> VDB
    VDB --> VERIFY
    VERIFY --> EXPLANATION
    VERIFY --> QUIZ
```

---

## ✨ Features

- **ABCD framework** — Anchor-Bridge-Construct-Deconstruct methodology
- **First-principles breakdown** — reduce complex topics to fundamentals
- **Concept mapping** — visual graph of concept dependencies
- **Comprehension check** — auto-generated quizzes from document content
- **Streamlit UI** — interactive tutoring interface

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run aegis/app.py
```

---

## 📁 Project Structure

```
ProjectAegis/
├── aegis/
│   ├── app.py             # Streamlit UI
│   ├── graph.py           # LangGraph pipeline
│   ├── parser.py          # Document structure parsing
│   ├── deconstruct.py     # ABCD framework logic
│   ├── explain.py         # First-principles explanation
│   ├── quiz.py            # Comprehension verification
│   └── store.py           # Vector persistence
├── tests/
└── README.md
```

---

## 📄 License

MIT © Md Sadman Bin Masud
