# ⚡ VentiRE : Copilote RAG Intelligent pour la Réglementation RE2020 & Ventilation

**VentiRE** est un système de génération augmentée par récupération (**RAG - Retrieval-Augmented Generation**) conçu de manière industrielle pour automatiser l'analyse, le croisement et l'interrogation de corpus réglementaires complexes dans le secteur du bâtiment.

Développé en **Python 3.12** autour du framework **LlamaIndex**, ce projet résout un défi majeur en ingénierie thermique et environnementale : permettre à un opérateur ou un auditeur technique d'interroger simultanément et sans ambiguïté des textes cadres, des exigences de contrôle terrain et des arrêtés historiques, tout en préservant la mise en forme et l'intégrité des tableaux de données ainsi que des grilles d'audit complexes.

---

## 🏗️ 1. Architecture et Pipeline de Données Multi-Documents

Le système est structuré autour d'un pipeline modulaire, garantissant une séparation stricte entre l'ingestion, la vectorisation, et l'interface utilisateur.


### Étape 1 : Ingestion Vision-to-Text (`LlamaParse`)
La documentation réglementaire se caractérise par une densité critique de tableaux croisés et de listes imbriquées. Les extracteurs PDF bruts standards détruisent souvent la linéarité de ces structures. VentiRE intègre l'API sémantique basée sur la vision de **LlamaParse** pour convertir les PDF en Markdown natif, transformant les tableaux complexes en grilles de texte propres et lisibles.

### Étape 2 : Stratégie d'Indexation Globale Unifiée
Pour permettre le croisement des informations, VentiRE utilise une approche multi-documents centralisée. Il s'appuie sur le `MarkdownNodeParser` pour découper le texte selon la hiérarchie logique des titres (`#`, `##`, `###`), garantissant que chaque fragment de texte (Node) conserve son contexte d'origine. Les embeddings globaux sont calculés via le moteur **FastEmbed** avec le modèle de pointe **`BAAI/bge-large-en-v1.5`** pour s'affranchir des limitations d'API externes et garantir un alignement sémantique parfait entre l'indexation et la recherche.

### Étape 3 : Moteur de Restitution Hybride & Reranking
L'interface de discussion bâtie sous **Streamlit** interroge l'index vectoriel permanent sauvegardé sur le disque. Pour éviter le "bruit" sémantique lorsque plusieurs réglementations utilisent des termes proches, le système couple la recherche vectorielle à un module **Cohere Rerank**. Ce re-classeur trie les fragments extraits avant de les soumettre au modèle local **Mistral AI** (via Ollama), configuré avec une température de `0.0` pour garantir des réponses fiables, techniques, sans aucune hallucination et entièrement sourcées.

---

## 📁 2. Organisation du Dépôt

La structure de ce projet reflète les meilleures pratiques de développement logiciel en IA :

```text
VentiRE/
├── .cache/                 # Cache local contenant les poids du modèle d'embedding FastEmbed
├── .venv/                  # Environnement virtuel isolé Python 3.12
├── data_source/            # Bibliothèque de fichiers formalisés en Markdown (.md)
│   ├── guide_re2020_structure.md
│   ├── protocole_ventilation_re2020_v2.md
│   └── arrete_24_mars_1982.md
├── PDF_DIR/         # Répertoire sécurisé d'entrée pour les documents PDF originaux
├── storage_re2020/         # Base de données vectorielle locale permanente (fichiers JSON)
├── .env                    # Fichier de configuration des clés API secrètes (GitIgnored)
├── path_1_llamaparse.py    # Module N°1 : Traitement et extraction Markdown via LlamaParse
├── path_2_indexing.py      # Module N°2 : Ingestion globale, parsing hiérarchique et indexation FastEmbed
├── path_4.py               # Module N°3 : Moteur d'Analyse Hybride Multidoc, Reranking et génération locale
└── web_app.py              # Module N°4 : Interface utilisateur graphique sous Streamlit. 
