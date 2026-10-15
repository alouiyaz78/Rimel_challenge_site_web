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
DOSSIER_RAG = os.path.join(BASE_DIR, "data", "rag")
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
                    "metadata": {"tag": tag, "source": "Fiches Rimel", "auteur": "Base documentaire du Rimel Challenge",
                                 "citation": "", "titre": titre, "section": section.strip(), "partie": i + 1},
                })
    return chunks


# ==============================================================================
# 2 bis. SOURCES HISTORIQUES EXTERNES (data/rag/) — attribution obligatoire
# ==============================================================================

SOURCES_EXTERNES = {
    "archives_kaabori_menzel_jemil.md": {
        "tag": "archives_kaabori",
        "source": "Kaabori",
        "auteur": "M. Mohamed Hedi Kaabori (leplacartuel.com)",
        "citation": "Selon les recherches de M. Mohamed Hedi Kaabori (leplacartuel.com)",
    },
    "these_hbaieb_bizerte.md": {
        "tag": "these_hbaieb",
        "source": "Hbaieb",
        "auteur": "Mohamed Ali Hbaieb, thèse de doctorat (Université de Tunis / Lyon 2 UMR 5648)",
        "citation": "D'après la thèse de doctorat de Mohamed Ali Hbaieb (Université de Tunis / Lyon 2 UMR 5648)",
    },
}
FIN_PHRASE = re.compile(r"(?<=[.!?؟;:])\s+")


def _paquets(texte: str, taille: int = TAILLE_MAX_CHUNK) -> list[str]:
    """Regroupe les phrases d'un texte en blocs d'environ `taille` caractères."""
    blocs, courant = [], ""
    for phrase in FIN_PHRASE.split(texte):
        while len(phrase) > taille:  # phrase interminable (texte mal ponctué) : coupe sur un espace
            coupe = phrase.rfind(" ", 0, taille)
            coupe = coupe if coupe > taille // 2 else taille
            if courant:
                blocs.append(courant.strip()); courant = ""
            blocs.append(phrase[:coupe].strip()); phrase = phrase[coupe:]
        if courant and len(courant) + len(phrase) > taille:
            blocs.append(courant.strip()); courant = ""
        courant += phrase + " "
    if courant.strip():
        blocs.append(courant.strip())
    return [b for b in blocs if len(b) >= 60]  # ignore les débris (titres isolés, numéros)


def _sections(markdown: str, niveau: str) -> list[tuple[str, str]]:
    """Coupe un markdown sur les titres du niveau donné (« ## » ou « ### »)."""
    parties = re.split(rf"^{niveau} ", markdown, flags=re.M)[1:]
    return [(p.partition("\n")[0].strip(), p.partition("\n")[2]) for p in parties]


def _frequences_lignes_courtes(texte: str) -> dict[str, int]:
    frequence: dict[str, int] = {}
    for l in texte.splitlines():
        cle = l.strip()
        if cle and len(cle) < 25:
            frequence[cle] = frequence.get(cle, 0) + 1
    return frequence


def _nettoyer_kaabori(texte: str, frequence: dict[str, int]) -> str:
    """Retire le menu de navigation du site (lignes courtes répétées sur chaque page, comptées
    sur tout le fichier) et recolle les lignes coupées."""
    lignes = [l.rstrip() for l in texte.splitlines()]
    paragraphes, courant = [], ""
    for l in lignes:
        net = l.strip()
        if not net or net in ("|", "---") or net.startswith("**URL Source"):
            if courant:
                paragraphes.append(courant); courant = ""
            continue
        if len(net) < 25 and frequence.get(net, 0) >= 3:  # entrée de menu répétée sur chaque page
            continue
        if l[:1].isspace() and courant:  # ligne de continuation (indentée par l'extraction HTML)
            courant += " " + net
        else:
            if courant:
                paragraphes.append(courant)
            courant = net
    if courant:
        paragraphes.append(courant)
    return "\n\n".join(re.sub(r"\s+", " ", p) for p in paragraphes)


# Glyphes de translittération de l'arabe mal décodés par l'extraction PDF de la thèse
# (ex. « Manzil ¶m∏l » = Manzil Jmîl, « cAbderra™mån » = ʿAbderraḥmân, « T∏n¶a » = Tînja).
GLYPHES_TRANSLITTERATION = {"™": "h", "∏": "î", "å": "â", "¶": "j", "•": "J", "ß": "Gh", "”": "s", "≠": "z", "§": "d"}


def _nettoyer_pdf(texte: str) -> str:
    """Texte extrait du PDF de la thèse : supprime numéros de page et césures, recolle les lignes,
    décode les glyphes de translittération."""
    for glyphe, lettre in GLYPHES_TRANSLITTERATION.items():
        texte = texte.replace(glyphe, lettre)
    texte = re.sub(r"\bc([A-Z])", r"ʿ\1", texte)  # « cAbd » -> « ʿAbd » (ʿayn)
    lignes = [l.strip() for l in texte.splitlines()]
    lignes = [l for l in lignes if l and not re.fullmatch(r"[\d\s]+", l) and l != "---"]
    texte = " ".join(lignes)
    texte = re.sub(r"([a-zà-ÿ])- ([a-zà-ÿ])", r"\1\2", texte)  # césures « dévelop- pement »
    return re.sub(r"\s+", " ", texte).strip()


def _page_non_redactionnelle(texte: str) -> bool:
    """Sommaire, tables des figures, bibliographie : listes de titres et de références,
    qui pousseraient le modèle à tirer des conclusions que le texte ne fait pas."""
    if re.search(r"TABLE DES|BIBLIOGRAPHIE|SOMMAIRE|INDEX DES|SOURCES ET BIBLIO", texte):
        return True
    return len(re.findall(r"\bFig\. ?\d", texte)) >= 4 or len(re.findall(r"\bp\. ?\d", texte)) >= 6


def decouper_sources_externes() -> list[dict]:
    chunks = []
    for nom, meta in SOURCES_EXTERNES.items():
        chemin = os.path.join(DOSSIER_RAG, nom)
        if not os.path.exists(chemin):
            logger.warning("Source historique absente : %s", chemin)
            continue
        with open(chemin, encoding="utf-8") as f:
            markdown = f.read()
        if meta["source"] == "Kaabori":
            frequence = _frequences_lignes_courtes(markdown)
            sections = [(titre, _nettoyer_kaabori(corps, frequence), re.search(r"URL Source :\*\*\s*(\S+)", corps))
                        for titre, corps in _sections(markdown, "##")]
            sections = [(t, c, u.group(1) if u else "leplacartuel.com") for t, c, u in sections]
        else:
            sections = [(titre, _nettoyer_pdf(corps), "thèse, " + titre.replace("Extrait de la ", "").lower())
                        for titre, corps in _sections(markdown, "###")]
            exclues = [t for t, c, _ in sections if _page_non_redactionnelle(c)]
            if exclues:
                logger.info("Pages non rédactionnelles ignorées (%s) : %s", meta["source"], ", ".join(exclues))
            sections = [(t, c, r) for t, c, r in sections if t not in exclues]
        for titre, texte, reference in sections:
            for i, morceau in enumerate(_paquets(texte)):
                chunks.append({
                    "id": f"{meta['tag']}-{len(chunks):04d}",
                    "document": f"[{meta['source']}] {titre}\n{morceau}",
                    "metadata": {
                        "tag": meta["tag"], "source": meta["source"], "auteur": meta["auteur"],
                        "citation": meta["citation"], "titre": titre, "section": titre,
                        "reference": reference, "partie": i + 1,
                    },
                })
    return chunks


# ==============================================================================
# 3. API PUBLIQUE
# ==============================================================================

def initialiser_base() -> int:
    """(Ré)indexe entièrement la base de connaissances. Retourne le nombre de chunks."""
    with open(FICHIER_CONNAISSANCES, encoding="utf-8") as f:
        chunks = decouper_fiches(f.read())
    chunks += decouper_sources_externes()
    if not chunks:
        raise ValueError(f"Aucune fiche exploitable dans {FICHIER_CONNAISSANCES}")

    client, _ = _ressources()
    try:
        client.delete_collection(NOM_COLLECTION)
    except Exception:
        pass  # première indexation
    collection = _collection(creer=True)
    for debut in range(0, len(chunks), 200):  # lots raisonnables pour le calcul des embeddings
        lot = chunks[debut:debut + 200]
        collection.add(
            ids=[c["id"] for c in lot],
            documents=[c["document"] for c in lot],
            metadatas=[c["metadata"] for c in lot],
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
    # Les sources historiques (~200 extraits) peuvent évincer les fiches Rimel, plus courtes et plus
    # ciblées : on ajoute toujours les 3 meilleures fiches si elles ne sont pas déjà dans les résultats.
    if not filtre_tag:
        deja = set(resultats["documents"][0])
        fiches = collection.query(query_texts=[query], n_results=3, where={"source": "Fiches Rimel"})
        for doc, meta, dist in zip(fiches["documents"][0], fiches["metadatas"][0], fiches["distances"][0]):
            if doc not in deja:
                resultats["documents"][0].append(doc); resultats["metadatas"][0].append(meta); resultats["distances"][0].append(dist)
    documents = resultats["documents"][0]
    if not documents:
        return "Aucun extrait pertinent trouvé dans les archives du Remel."

    extraits = []
    for doc, meta, distance in zip(documents, resultats["metadatas"][0], resultats["distances"][0]):
        entete = f"[SOURCE : {meta.get('auteur', meta['tag'])} | {meta.get('reference', meta.get('section', ''))} | pertinence {1 - distance:.2f}]"
        if meta.get("citation"):
            entete += f"\n[CITATION OBLIGATOIRE : « {meta['citation']}… »]"
        extraits.append(f"{entete}\n{doc}")
    return "\n\n---\n\n".join(extraits)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"{initialiser_base()} chunks indexés.")
