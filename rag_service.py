"""Base de connaissances vectorielle (RAG local) sur le patrimoine du Remel.

Source : docs/remel_territory_knowledge.md, découpé en fiches (## ...) puis en
sections (### ...). Chaque chunk est indexé dans ChromaDB (stockage persistant
data/vector_db/) avec les métadonnées `tag`, `titre` et `section`.

Usage :
    python -c "import rag_service; rag_service.initialiser_base()"
    rag_service.rechercher_contexte("Quel est le rôle du Saccharum dans les dunes ?")
"""

import logging
import os
import re
import threading

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FICHIER_CONNAISSANCES = os.path.join(BASE_DIR, "docs", "remel_territory_knowledge.md")
DOSSIER_BASE = os.path.join(BASE_DIR, "data", "vector_db")
NOM_COLLECTION = "remel_patrimoine"
MODELE_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Au-delà, une section est redécoupée par paragraphes (le modèle tronque à 128 tokens ≈ 500 caractères).
TAILLE_MAX_CHUNK = 500

_verrou = threading.Lock()
_client = None
_fonction_embeddings = None


# ==============================================================================
# 1. ACCÈS CHROMADB (initialisation paresseuse, thread-safe)
# ==============================================================================

def _ressources():
    global _client, _fonction_embeddings
    with _verrou:
        if _client is None:
            os.makedirs(DOSSIER_BASE, exist_ok=True)
            _client = chromadb.PersistentClient(path=DOSSIER_BASE)
            _fonction_embeddings = SentenceTransformerEmbeddingFunction(model_name=MODELE_EMBEDDINGS)
    return _client, _fonction_embeddings


def _collection(creer: bool = False):
    client, fonction = _ressources()
    if creer:
        return client.get_or_create_collection(
            NOM_COLLECTION,
            embedding_function=fonction,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(NOM_COLLECTION, embedding_function=fonction)


# ==============================================================================
# 2. DÉCOUPAGE DU MARKDOWN EN CHUNKS SÉMANTIQUES
# ==============================================================================

def _redecouper(texte: str) -> list[str]:
    """Regroupe les paragraphes d'une section trop longue en blocs <= TAILLE_MAX_CHUNK."""
    if len(texte) <= TAILLE_MAX_CHUNK:
        return [texte]
    blocs, courant = [], ""
    for paragraphe in re.split(r"\n\s*\n", texte):
        if courant and len(courant) + len(paragraphe) > TAILLE_MAX_CHUNK:
            blocs.append(courant.strip())
            courant = ""
        courant += paragraphe + "\n\n"
    if courant.strip():
        blocs.append(courant.strip())
    return blocs


def decouper_fiches(markdown: str) -> list[dict]:
    """Découpe le fichier de connaissances.

    Format attendu :
        ## Fiche 1 — Titre de la fiche
        tag: histoire_foret
        ### Sous-section
        texte...
    """
    chunks = []
    for bloc_fiche in re.split(r"^## ", markdown, flags=re.M)[1:]:
        entete, _, corps = bloc_fiche.partition("\n")
        titre = re.sub(r"^Fiche\s*\d+\s*[—–-]\s*", "", entete).strip()
        tag_trouve = re.search(r"^tag:\s*(\S+)", corps, flags=re.M)
        if not tag_trouve:
            logger.warning("Fiche sans tag ignorée : %s", titre)
            continue
        tag = tag_trouve.group(1)
        corps = re.sub(r"^(tag|sources?):.*$", "", corps, flags=re.M)

        for bloc_section in re.split(r"^### ", corps, flags=re.M):
            section, _, texte = bloc_section.partition("\n")
            texte = texte.strip()
            if not texte:
                continue
            for i, morceau in enumerate(_redecouper(texte)):
                chunks.append({
                    "id": f"{tag}-{len(chunks):03d}",
                    # Le titre est répété dans le document pour enrichir l'embedding.
                    "document": f"{titre} — {section.strip()}\n{morceau}",
                    "metadata": {"tag": tag, "titre": titre, "section": section.strip(), "partie": i + 1},
                })
    return chunks


# ==============================================================================
# 3. API PUBLIQUE
# ==============================================================================

def initialiser_base() -> int:
    """(Ré)indexe entièrement la base de connaissances. Retourne le nombre de chunks."""
    with open(FICHIER_CONNAISSANCES, encoding="utf-8") as f:
        chunks = decouper_fiches(f.read())
    if not chunks:
        raise ValueError(f"Aucune fiche exploitable dans {FICHIER_CONNAISSANCES}")

    client, _ = _ressources()
    try:
        client.delete_collection(NOM_COLLECTION)
    except Exception:
        pass  # première indexation
    collection = _collection(creer=True)
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["document"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    logger.info("%d chunks indexés dans la collection %s", len(chunks), NOM_COLLECTION)
    return len(chunks)


def rechercher_contexte(query: str, n_results: int = 3, filtre_tag: str | None = None) -> str:
    """Retourne les extraits les plus pertinents, formatés pour être injectés dans un prompt."""
    try:
        collection = _collection()
    except Exception:
        logger.info("Collection %s absente : indexation initiale.", NOM_COLLECTION)
        initialiser_base()
        collection = _collection()

    resultats = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"tag": filtre_tag} if filtre_tag else None,
    )
    documents = resultats["documents"][0]
    if not documents:
        return "Aucun extrait pertinent trouvé dans les archives du Remel."

    extraits = []
    for doc, meta, distance in zip(documents, resultats["metadatas"][0], resultats["distances"][0]):
        extraits.append(f"[{meta['tag']} | pertinence {1 - distance:.2f}]\n{doc}")
    return "\n\n---\n\n".join(extraits)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"{initialiser_base()} chunks indexés.")
