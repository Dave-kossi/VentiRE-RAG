# VentiRE-RAG
Assistant RAG intelligent et local spécialisé dans l'analyse croisée de la réglementation RE2020 et des protocoles de contrôle de la ventilation. Développé avec LlamaIndex, FastEmbed et Streamlit.

# ⚡ VentiRE : Assistant RAG Intelligent pour la Réglementation RE2020 & Ventilation

**VentiRE** est un système de génération augmentée par récupération (**RAG - Retrieval-Augmented Generation**) conçu de manière industrielle pour automatiser l'analyse, le croisement et l'interrogation de corpus réglementaires complexes dans le secteur du bâtiment.

Développé en **Python 3.12** autour du framework **LlamaIndex**, ce projet résout un défi majeur en ingénierie thermique et environnementale : permettre à un opérateur ou un auditeur technique d'interroger simultanément et sans ambiguïté des textes cadres (comme le *Guide Général RE2020*) et des exigences de contrôle terrain ultra-spécifiques (comme le *Protocole Ventilation RE2020*), tout en préservant la mise en forme et l'intégrité des tableaux de données ainsi que des grilles d'audit complexes.

---

## 🏗️ 1. Architecture et Pipeline de Données

Le système est structuré autour d'un pipeline modulaire, garantissant une séparation stricte entre l'ingestion, la vectorisation, et l'interface utilisateur.
