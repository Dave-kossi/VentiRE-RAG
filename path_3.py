import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# 1. Imports de base du Framework RAG
from llama_index.core import StorageContext, VectorStoreIndex, Settings

# 2. Imports pour la stratégie de Retrieval Hybride et Fusion
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

# 3. Import du Query Engine (C'est cet import qui manquait !)
from llama_index.core.query_engine import RetrieverQueryEngine

# 4. Import du Reranker de Cohere
from llama_index.postprocessor.cohere_rerank import CohereRerank

# 5. Imports pour l'infrastructure Locale (Embedding gratuit & Faux LLM de validation)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import MockLLM
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

print("--- [R&D] Initialisation du modèle d'embedding local BGE-M3 (Gratuit) ---")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# CORRECTION ICI : Au lieu de None, on met un faux LLM local
Settings.llm = MockLLM(max_tokens=20)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return BASE_DIR / candidate


def setup_advanced_query_engine(storage_dir: str | Path = "./storage_re2020"):
    print("--- Configuration du moteur de recherche hybride ---")
    storage_dir = resolve_path(storage_dir)
    if not storage_dir.exists():
        raise FileNotFoundError(f"Dossier de stockage introuvable : {storage_dir}")
    
    # 1. Reconstitution du stockage local
    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    
    # Chargement de l'index vectoriel local
    index = VectorStoreIndex(nodes=[], storage_context=storage_context)
    
    # 2. Branche Sémantique Locale (Dense) - Cherche parmi les enfants
    vector_retriever = index.as_retriever(similarity_top_k=10)
    
    # 3. Branche Mots-clés (Sparse BM25) - Idéal pour les acronymes comme Bbio, DH, Cep
    nodes = list(storage_context.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    
    # 4. Fusion des deux stratégies (Hybrid Search)
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=10,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True
    )
    
    # 5. Post-traitement : Reranker de Cohere (Optionnel mais recommandé pour le top-k)
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    node_postprocessors = []
    
    if not cohere_api_key:
        print("💡 Note : COHERE_API_KEY manquante. Le Reranker de précision est désactivé.")
    else:
        reranker = CohereRerank(api_key=cohere_api_key, top_n=3, model="rerank-multilingual-v3.0")
        node_postprocessors.append(reranker)
        
    # 6. Assemblage du moteur de requêtage
    # Note : Si tu n'as pas de clé LLM, nous allons configurer le moteur pour renvoyer 
    # UNIQUEMENT les blocs de texte (sources) bruts récupérés pour éviter l'erreur LLM OpenAI.
    # Pour cela, on demande à LlamaIndex d'agir en mode "retriever pur" ou on met un LLM fictif.
    from llama_index.core.llms import MockLLM
    Settings.llm = MockLLM(max_tokens=20) # Évite les appels OpenAI sous le capot
    
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=node_postprocessors
    )
    
    return query_engine, hybrid_retriever


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teste la récupération hybride dans un index local.")
    parser.add_argument(
        "query",
        nargs="?",
        default="Quelles sont les valeurs limites du compteur DH pour le confort d'été ?",
        help="Question de test.",
    )
    parser.add_argument(
        "-s",
        "--storage-dir",
        default="./storage_re2020",
        help="Dossier de stockage généré par hiera.py.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=3,
        help="Nombre de blocs à afficher.",
    )
    return parser

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    storage_dir = resolve_path(args.storage_dir)
    
    if not storage_dir.exists():
        print(f"Erreur : Le dossier {storage_dir} n'existe pas. Lance d'abord le script d'indexation.")
    else:
        engine, retriever = setup_advanced_query_engine(storage_dir)
        query_test = args.query
        print(f"\n🚀 [Question Test R&D] : {query_test}")
        
        # Pour voir exactement ce que notre RAG "récupère" de manière transparente avant la génération :
        retrieved_nodes = retriever.retrieve(query_test)
        
        print("\n=== BLOCS DE CONTEXTE EXTRACTS (PARENTS INTERCONNECTÉS) ===")
        for idx, node in enumerate(retrieved_nodes[:args.top_k]):
            print(f"\n--- Document Source #{idx+1} (Score: {node.score:.4f}) ---")
            print(node.node.text[:500] + "...") # Affiche les 500 premiers caractères du bloc parent
        print("===========================================================\n")
