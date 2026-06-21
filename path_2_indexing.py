import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import MarkdownNodeParser
# On bascule sur FastEmbed qui est ultra-stable avec Pydantic V2 et très rapide sur CPU
from llama_index.embeddings.fastembed import FastEmbedEmbedding

load_dotenv()

# 1. Configuration du modèle d'embedding locale et robuste
print("--- [R&D] Initialisation du modèle d'embedding FastEmbed ---")
# Utilisation d'un excellent modèle de texte supporté nativement sans bug Pydantic
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5")
Settings.llm = None # Désactivé pour l'indexation pure

DATA_DIR = "./data_source"
STORAGE_DIR = "./storage_re2020"

def build_global_knowledge_base():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"❌ Erreur : Le dossier '{DATA_DIR}' est vide ou n'existe pas.")
        return

    print(f"📁 Chargement des documents depuis {DATA_DIR}...")
    
    # Charge tous les fichiers (.md) du dossier d'un coup
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    print(f"📄 {len(documents)} fichiers sources détectés.")

    # 2. Découpage / Parsing
    print("🧩 Découpage structurel en cours (Markdown Parsing)...")
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print(f"⚡ {len(nodes)} nœuds textuels générés.")

    # 3. Création de la Base de Connaissances Vectorielle
    print("🧠 Génération des vecteurs (Embeddings)...")
    storage_context = StorageContext.from_defaults()
    
    # On construit l'index global unifié
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    # 4. Sauvegarde locale permanente
    print(f"💾 Sauvegarde de la base de données dans '{STORAGE_DIR}'...")
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print("✅ Indexation réussie ! Tous vos documents sont liés et prêts.")

if __name__ == "__main__":
    build_global_knowledge_base()