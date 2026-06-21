import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from llama_index.core import StorageContext, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.llms.ollama import Ollama

# Utilisation exclusive de FastEmbed pour éviter les conflits Pydantic v2
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# 1. Chargement des clés API et de l'environnement (.env)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

print("--- [R&D] Initialisation des composants locaux (FastEmbed BGE-M3 & Mistral) ---")
# CORRECTION : Nom du modèle adapté pour FastEmbed et suppression de l'argument 'cache_folder' problématique
Settings.embed_model = FastEmbedEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 2. Configuration du LLM Français via Ollama (Température 0.0 pour un max de rigueur)
Settings.llm = Ollama(model="mistral", request_timeout=500.0, temperature=0.0)

def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return BASE_DIR / candidate


def build_qa_template(document_name: str = "RE2020") -> PromptTemplate:
    template_rag_francais = (
        "Vous êtes un ingénieur expert chargé d'analyser un document technique ou réglementaire.\n"
        f"Document de référence : {document_name}\n"
        "Voici les extraits véridiques issus de ce document :\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "En vous basant uniquement sur les extraits ci-dessus, répondez de manière précise et technique "
        "à la question suivante. Si l'information n'est pas présente dans le contexte, dites humblement "
        "que vous ne savez pas, n'inventez rien.\n"
        "Question : {query_str}\n"
        "Réponse détaillée en français :"
    )
    return PromptTemplate(template_rag_francais)


def run_french_rag_system(
    query: str,
    storage_dir: str | Path = "./storage_re2020",
    document_name: str = "RE2020",
):
    storage_dir = resolve_path(storage_dir)
    if not storage_dir.exists():
        raise FileNotFoundError(f"Dossier de stockage introuvable : {storage_dir}")

    # Reconstitution du stockage local généré à l'étape 2 (Structure Parent-Enfant)
    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    index = VectorStoreIndex(nodes=[], storage_context=storage_context)
    
    # 4. Retrieval Hybride (Dense Search + Sparse Search)
    vector_retriever = index.as_retriever(similarity_top_k=10)
    nodes = list(storage_context.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=10,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True
    )
    
    # 5. Couche de Reranking Cross-Encoder via Cohere (si clé présente)
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    node_postprocessors = []
    if cohere_api_key:
        reranker = CohereRerank(api_key=cohere_api_key, top_n=3, model="rerank-multilingual-v3.0")
        node_postprocessors.append(reranker)
        
    # Assemblage du Query Engine avec notre Prompt personnalisé
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=node_postprocessors,
        text_qa_template=build_qa_template(document_name)
    )
    
    print(f"\n [Question] : {query}")
    print("-> Mistral analyse les sources locales et rédige la synthèse réglementaire...\n")
    
    # Exécution du RAG complet
    response = query_engine.query(query)
    return response


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interroge un index documentaire local avec le RAG.")
    parser.add_argument(
        "query",
        nargs="?",
        default="Quelles sont les valeurs limites du compteur DH pour le confort d'été ?",
        help="Question à poser au document.",
    )
    parser.add_argument(
        "-s",
        "--storage-dir",
        default="./storage_re2020",
        help="Dossier de stockage généré par hiera.py.",
    )
    parser.add_argument(
        "-d",
        "--document-name",
        default="RE2020",
        help="Nom affiché dans le prompt de réponse.",
    )
    return parser

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    storage_dir = resolve_path(args.storage_dir)
    
    if storage_dir.exists():
        rag_response = run_french_rag_system(args.query, storage_dir, args.document_name)
        
        print(f"=== RÉPONSE DE MISTRAL (RAG {args.document_name}) ===")
        print(rag_response)
        print("========================================\n")
    else:
        print(f"Erreur : Le dossier {storage_dir} n'existe pas. Relance d'abord l'indexation.")
