import os
import streamlit as st
from path_4 import run_french_rag_system

# Configuration de la page Web (Positionnement Copilote)
st.set_page_config(
    page_title="VentiRE - Copilote RE2020 & Ventilation",
    page_icon="⚡",
    layout="centered"
)

# Style et titre de l'application
st.title("⚡ VentiRE")
st.markdown("### Votre Copilote Réglementaire : RE2020, Ventilation & Arrêtés")
st.caption("Aide à la décision et recherche croisée : Guides RE2020, Protocoles de vérification VMC & Arrêté du 24 mars 1982")
st.divider()

STORAGE_DIR = "./storage_re2020"

# Initialisation de l'historique de discussion dans la session Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": (
                "Bonjour ! Je suis **VentiRE**, votre copilote réglementaire.\n\n"
                "Je suis conçu pour vous assister dans vos recherches techniques et croiser rapidement les exigences des documents suivants :\n"
                "- 🌡️ Le confort d'été (Seuils DH, Bbio, Cep...)\n"
                "- 🌀 Les protocoles de vérification des systèmes de ventilation\n"
                "- 📏 Les débits minimaux d'extraction d'air réglementaires (Arrêté du 24 mars 1982)\n\n"
                "Sur quel point réglementaire ou technique souhaitez-vous que je vous assiste ?"
            )
        }
    ]

# Affichage des messages historiques à chaque rechargement de l'écran
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie utilisateur
if query := st.chat_input("Posez votre question à votre copilote (ex: Quelle est la tolérance des mesures de pression ?)..."):
    
    # 1. Afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
        
    # 2. Générer la réponse du RAG avec un indicateur visuel de chargement
    with st.chat_message("assistant"):
        with st.spinner("🧠 VentiRE parcourt le corpus documentaire (Recherche hybride & Synthèse Mistral)..."):
            try:
                if os.path.exists(STORAGE_DIR):
                    rag_response = run_french_rag_system(query, STORAGE_DIR)
                    response_text = str(rag_response)
                else:
                    response_text = "❌ Erreur : Le dossier de stockage local `./storage_re2020` est introuvable. Veuillez relancer l'indexation (path_2_indexing.py)."
            except Exception as e:
                response_text = f"❌ Une erreur technique est survenue lors de l'analyse : {e}"
        
        # Affichage du résultat final
        st.write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})