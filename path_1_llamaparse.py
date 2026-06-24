import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# 1. Chargement de la clé API LLAMA_CLOUD_API_KEY depuis ton fichier .env
load_dotenv()

def parse_re2020_document(file_path: str, output_md_path: str):
    print(f"--- Début du parsing de : {file_path} ---")
    
    # 2. Configuration du parseur de LlamaIndex (Layout-Aware)
    # result_type="markdown" est indispensable pour reconstruire les tableaux de VMC en grilles MD
    parser = LlamaParse(
        result_type="markdown",  # Format de sortie requis pour l'indexation
        num_workers=4,           # Utilise la parallélisation pour accélérer le traitement
        verbose=True,
        language="fr"            # Optimisé pour le protocole de ventilation en français
    )
    
    # 3. Association du parseur à l'extension PDF via SimpleDirectoryReader
    file_extractor = {".pdf": parser}
    
    reader = SimpleDirectoryReader(
        input_files=[file_path],
        file_extractor=file_extractor
    )
    
    # 4. Envoi au Cloud LlamaParse et récupération des données structurées
    documents = reader.load_data()
    print(f"Extraction réussie ! Nombre de sections extraites : {len(documents)}")
    
    # 5. Sauvegarde du rendu Markdown dans le dossier cible
    with open(output_md_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.text)
            f.write("\n\n--- PAGE BREAK ---\n\n")
            
    print(f"Fichier Markdown de contrôle sauvegardé sous : {output_md_path}")
    return documents

if __name__ == "__main__":
    # --- CONFIGURATION ADAPTÉE À TON DOSSIER PDF_DIR ---
    PDF_INPUT = "./PDF_DIR/Arrete2-03-1982.pdf"
    MD_OUTPUT = "./data_source/Arrete2-03-1982.md"
    
    # Vérification de la clé API
    if not os.environ.get("LLAMA_CLOUD_API_KEY"):
        print("❌ Erreur : Veuillez définir la variable LLAMA_CLOUD_API_KEY dans votre fichier .env")
    else:
        # Création automatique des dossiers de l'architecture s'ils n'existent pas
        os.makedirs("./PDF_DIR", exist_ok=True)
        os.makedirs("./data_source", exist_ok=True)
        
        # Vérification de la présence du fichier
        if os.path.exists(PDF_INPUT):
            parsed_docs = parse_re2020_document(PDF_INPUT, MD_OUTPUT)
            print("\n🚀 Étape 1/2 validée avec succès avec LlamaParse !")
            print("Vous pouvez maintenant lancer : python path_2_indexing.py pour mettre à jour la base de connaissances.")
        else:
            print(f"❌ Erreur : Le fichier '{PDF_INPUT}' est introuvable.")
            print(f"👉 Assurez-vous d'avoir bien déposé le fichier 'Arrete2-03-1982.pdf' dans le dossier './PDF_DIR/'.")
