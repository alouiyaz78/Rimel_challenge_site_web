import json
import os
import re
import threading
import time
import unicodedata
import uuid
from collections import OrderedDict, deque

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
from flask import jsonify
from supabase import create_client, Client

import stats_inscriptions
from agents import BASE_DIR, ReponseIndisponible, message_surcharge, orchestrer, theme_indicatif

app = Flask(__name__)

# ==============================================================================
# 1. RATE LIMITING PAR IP (Flask-Limiter)
# ==============================================================================
def ip_visiteur() -> str:
    """IP réelle du visiteur. Derrière le tunnel Cloudflare (cloudflared), toutes les requêtes
    arrivent de 127.0.0.1 : sans cela, tous les visiteurs partageraient le même quota.
    L'en-tête CF-Connecting-IP n'est cru que s'il vient du tunnel local (sinon falsifiable)."""
    adresse = get_remote_address()
    if adresse in ("127.0.0.1", "::1"):
        return request.headers.get("CF-Connecting-IP", adresse)
    return adresse


limiter = Limiter(
    key_func=ip_visiteur,
    app=app,
    storage_uri="memory://",
)


@app.errorhandler(429)
def trop_de_requetes(e):
    return jsonify({
        "reply": "Doucement, coureur ! 🏃 Vous avez envoyé beaucoup de questions. "
                 "Merci de patienter une minute avant de réessayer."
    }), 429


# ==============================================================================
# 2. FAQ LOCALE & REPONSES COMMUNES (zéro coût LLM)
# ==============================================================================

def _normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", texte.lower())
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", texte)).strip()


REPONSES_COMMUNES = {
    # Secours, Sécurité & Urgence
    "ambulance": "Le dispositif médical sur site est assuré par CAP Ambulance (équipements catégorie A et équipe médicale qualifiée). En cas d'urgence en forêt, composez également le 198 (Protection Civile).",
    "secours": "La sécurité est assurée conjointement par CAP Ambulance (équipements catégorie A sur le terrain) et la Protection Civile (198).",
    "urgence": "En cas d'urgence sur le parcours ou dans la forêt de Rimel, contactez immédiatement la Protection Civile au 198 ou adressez-vous aux postes de secours CAP Ambulance (équipements catégorie A).",
    "198": "Le 198 est le numéro d'urgence officiel de la Protection Civile en Tunisie.",

    # Écologie, Mascotte & Valeurs
    "chardonneret": "Le chardonneret (طير الحسون) est le symbole de l'édition 2026 — c'est l'oiseau du visuel. Il sensibilise contre sa capture et sa vente illégale en cage : en Tunisie, 74 espèces d'oiseaux sur 180 sont menacées.",
    "oiseau": "L'oiseau du visuel est le chardonneret (طير الحسون), symbole de l'édition 2026 : il sensibilise contre sa capture et sa vente illégale en cage. En Tunisie, 74 espèces d'oiseaux sur 180 sont menacées.",
    "ecologie": "Initiative Net Zero Trail 2026 🌳 : le 12 avril 2026, 3 000 arbres plantés sur 2 hectares en forêt de Nadhour (avec Ecopact, la DGF et Natilait) ; le 1er août 2026, prévention des incendies avec le WWF North Africa et la DGF. Sur la course : zéro gobelet jetable, gobelet réutilisable obligatoire.",
    "gobelet": "Le gobelet réutilisable personnel est STRICTEMENT OBLIGATOIRE sous peine de disqualification ♻️. Aucun gobelet jetable n'est fourni sur les ravitaillements (démarche Net Zero Trail). Emballages à conserver sur soi.",
    "jetable": "Les gobelets jetables sont strictement interdits dans le cadre du Net Zero Trail. Vous devez obligatoirement courir avec votre propre gobelet réutilisable.",
    "natilait": "Natilait est le partenaire titre historique du Rimel Challenge, engagé fidèlement depuis 5 ans aux côtés des coureurs et de la préservation de la nature.",

    # Courses & Points ITRA
    "kids": "La course Kids 1 km, baptisée 'Course des Super Héros', est ouverte à tous les enfants (10 DT) pour découvrir le trail en toute sécurité dans la forêt de Rimel.",
    "super heros": "La 'Course des Super Héros' est le format Kids de 1 km dédié aux enfants (tarif : 10 DT) pour s'initier aux joies du trail.",
    "itra": "Points ITRA officiels attribués : 1 point ITRA pour le Trail 22 km et 2 points ITRA pour le grand défi de l'Ultra-Trail 50 km.",
    "50 km": "L'Ultra-Trail 50 km est le grand défi de l'édition : 100 DT (dossard + t-shirt officiel inclus), attribue 2 points ITRA. Départ à 07h00.",
    "22 km": "Le Trail 22 km (35 DT) est chronométré avec puce électronique et attribue 1 point ITRA. Départ groupé à 08h30.",
    "10 km": "Deux formats 10 km sont proposés : Trail Découverte sans puce (15 DT) et Mini-Trail chronométré avec puce (20 DT). Départ à 08h30.",

    # Logistique, Tarifs & Localisation
    "we run rimel": "L'association locale We Run Rimel et la Municipalité de Menzel Jemil collaborent étroitement à l'organisation et au balisage du Rimel Challenge.",
    "heberg": "L'organisation propose une formule officielle Hébergement + petit-déjeuner à 30 DT (réservable sur Sportsense.tn). Des hôtels sont également disponibles sur la corniche de Bizerte (~10 min).",
    "navette": "🚌 Navette depuis Tunis (Bizerta Top Travel) pour le 10 km, le 22 km et les accompagnateurs : 200 places assises uniquement, réservation obligatoire (option à cocher lors de l'inscription sur Sportsense, 10 DT). Aller : départ à 06h00 devant le Ministère du Tourisme (Avenue Habib Bourguiba, Tunis). Retour : départ à 13h00 depuis la forêt de Rimel. Coureurs du 50 km : un microbus dédié est spécifiquement prévu pour les coureurs du 50 km (horaires adaptés au départ matinal et aux arrivées tardives, ajusté en fonction des réservations).",
    "horaire": "Horaires officiels du Jour J : 05h30 ouverture de la table d'accueil au Village de Rimel (derniers paiements et dossards), 07h00 départ du 50 km, 08h15 briefing, 08h30 départ groupé 10 km et 22 km, 08h45 départ de la Course des Super Héros Kids (1 km), 16h30 cérémonie de remise des prix, 18h00 clôture du village.",


}


# Questions de préparation : elles citent souvent une distance ("plan pour le 50 km")
# mais doivent aller au Coach, pas à la fiche tarif de la FAQ.
INTENTIONS_COACH = (
    "entrain", "plan", "prepar", "allure", "rythme", "cadence", "combien de temps", "temps pour",
    "chrono", "hydrat", "boire", "eau", "electrolyte", "sels", "nutrition", "manger", "chaussure",
    "guetre", "equipement", "materiel", "conseil", "debutant", "niveau", "semaine", "fractionne",
    "cote", "sortie longue", "ppg", "renforcement", "blessure", "recuperation", "marche", "cyrano",
    # Visite et patrimoine : réponse détaillée de l'Ambassadeur (RAG), pas la fiche FAQ « menzel »
    "visit", "circuit", "station", "monument", "mosquee", "zaouia", "histoire", "patrimoine",
    # Tracés et profils : réponse détaillée de l'agent Parcours, pas la fiche tarif
    "denivel", "profil", "trace", "altitude", "barriere", "ravitaill", "gpx", "difficulte",
)


DEPART_COURSES = (
    "Toutes les courses de trail (50 km, 22 km, 10 km et Kids) partent du Village départ en pleine forêt de Rimel, "
    "sur l'esplanade d'animation près du Centre des stages et des vacances de Rimel (مركز التربصات والإصطياف والتخييم بالرمال). "
    "Table d'accueil ouverte dès 05h30."
)
DEPART_NUANCE = (
    "Il y a deux départs distincts 📍 : les COURSES DE TRAIL (50, 22, 10 km et Kids) partent du Village départ en forêt de Rimel, "
    "sur l'esplanade près du Centre des stages et des vacances de Rimel ; le CIRCUIT CULTUREL (visite à pied des 16 stations) "
    "part de la station de bus au centre-ville de Menzel Jemil. Lequel vous intéresse ?"
)
# Réponses testées AVANT les mots-clés généraux (sinon « dossard pour le 22 km » tomberait sur la fiche tarif du 22 km).
REPONSES_PRIORITAIRES = {
    # Jauge : testée avant « dossard » (« combien de dossards ? » porte sur le plafond, pas sur le retrait)
    "combien de dossards": "La limite maximale est strictement fixée à 3 500 participants au total, toutes courses confondues 🎽. Les inscriptions seront closes dès que ce plafond sera atteint : le nombre d'inscrits en direct est affiché en haut du site.",
    "combien de places": "La limite maximale est strictement fixée à 3 500 participants au total, toutes courses confondues 🎽. Les inscriptions seront closes dès que ce plafond sera atteint : le nombre d'inscrits en direct est affiché en haut du site.",
    "combien de participants": "La limite maximale est strictement fixée à 3 500 participants au total, toutes courses confondues 🎽. Les inscriptions seront closes dès que ce plafond sera atteint : le nombre d'inscrits en direct est affiché en haut du site.",
    "nombre de participants": "La limite maximale est strictement fixée à 3 500 participants au total, toutes courses confondues 🎽. Les inscriptions seront closes dès que ce plafond sera atteint : le nombre d'inscrits en direct est affiché en haut du site.",
    "nombre maximum": "La limite maximale est strictement fixée à 3 500 participants au total, toutes courses confondues 🎽. Les inscriptions seront closes dès que ce plafond sera atteint : le nombre d'inscrits en direct est affiché en haut du site.",
    "une limite": "La limite maximale est strictement fixée à 3 500 participants au total, toutes courses confondues 🎽. Les inscriptions seront closes dès que ce plafond sera atteint : le nombre d'inscrits en direct est affiché en haut du site.",
    "jauge": "La limite maximale est strictement fixée à 3 500 participants au total, toutes courses confondues 🎽. Les inscriptions seront closes dès que ce plafond sera atteint : le nombre d'inscrits en direct est affiché en haut du site.",
    "quota": "La limite maximale est strictement fixée à 3 500 participants au total, toutes courses confondues 🎽. Les inscriptions seront closes dès que ce plafond sera atteint : le nombre d'inscrits en direct est affiché en haut du site.",
    "navette": REPONSES_COMMUNES["navette"],
    "microbus": REPONSES_COMMUNES["navette"],
    "depuis tunis": REPONSES_COMMUNES["navette"],
    "dossard": "Retrait des dossards du 26 octobre au 3 novembre (ou jusqu'à épuisement des stocks), de 10h00 à 19h00, dans l'un des 5 points officiels : Peak Tunis City (Géant), Peak Azur City, Peak La Soukra, Foire de Bizerte, Bureau We Run Rimel (Menzel Jemil). Derniers retraits possibles le Jour J à la table d'accueil du Village de Rimel dès 05h30. Munissez-vous d'une pièce d'identité et de la décharge médicale / du certificat médical.",
    "preinscri": "La préinscription en ligne sur Sportsense ne suffit pas ⚠️ : votre dossier reste en attente tant que le paiement n'est pas effectué. Validez-le du 26 octobre au 3 novembre (10h00–19h00) dans l'un des 5 points de retrait (Peak Tunis City (Géant), Peak Azur City, Peak La Soukra, Foire de Bizerte, Bureau We Run Rimel (Menzel Jemil)), ou le Jour J à la table d'accueil du Village de Rimel dès 05h30 (pièce d'identité + décharge médicale / certificat).",
}
MOTS_COURSE = re.compile(r"\b(km|trail|course|courses|kids|ultra|mini|semi|coureurs?)\b")
QUESTION_DE_LIEU = re.compile(r"\b(ou|lieu|adresse|trouve|localis|rendez vous)")


def reponse_faq(question: str) -> str | None:
    q = _normaliser(question)
    if any(re.search(rf"\b{intention}", q) for intention in INTENTIONS_COACH):
        return None
    # « Où est le départ ? » : lieu de départ sportif ou culturel (pas l'heure de départ)
    if re.search(r"\bdepart", q) and QUESTION_DE_LIEU.search(q):
        return DEPART_COURSES if MOTS_COURSE.search(q) else DEPART_NUANCE
    for cle, reponse in REPONSES_PRIORITAIRES.items():
        if cle in q:
            return reponse
    for cle, reponse in REPONSES_COMMUNES.items():
        cle_norm = _normaliser(cle)
        if cle_norm in q:
            return reponse
    return None


# ==============================================================================
# 3. CACHE LRU HORODATÉ (questions identiques déjà posées)
# ==============================================================================

class CacheTTL:
    """Cache LRU thread-safe avec expiration, pour ne pas servir indéfiniment une réponse."""

    def __init__(self, taille_max: int = 500, ttl_secondes: int = 3600):
        self.taille_max = taille_max
        self.ttl = ttl_secondes
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, cle: str) -> str | None:
        with self._lock:
            entree = self._data.get(cle)
            if entree is None:
                return None
            horodatage, valeur = entree
            if time.monotonic() - horodatage > self.ttl:
                del self._data[cle]
                return None
            self._data.move_to_end(cle)
            return valeur

    def set(self, cle: str, valeur: str) -> None:
        with self._lock:
            self._data[cle] = (time.monotonic(), valeur)
            self._data.move_to_end(cle)
            while len(self._data) > self.taille_max:
                self._data.popitem(last=False)


cache_reponses = CacheTTL()


# ==============================================================================
# 3 bis. MÉMOIRE DE SESSION (historique des échanges par conversation)
# ==============================================================================

class HistoriqueSessions:
    """Historique en mémoire, par session_id, avec expiration et nombre de sessions borné.

    Stockage local au processus : avec plusieurs workers (gunicorn -w N), il faudrait
    un stockage partagé (Redis…) pour qu'une conversation garde son contexte.
    """

    def __init__(self, max_sessions: int = 5000, max_messages: int = 12, ttl_secondes: int = 7200):
        self.max_sessions = max_sessions
        self.max_messages = max_messages
        self.ttl = ttl_secondes
        self._data: OrderedDict[str, tuple[float, deque]] = OrderedDict()
        self._lock = threading.Lock()

    def lire(self, session_id: str) -> list[dict]:
        with self._lock:
            entree = self._data.get(session_id)
            if entree is None or time.monotonic() - entree[0] > self.ttl:
                self._data.pop(session_id, None)
                return []
            return list(entree[1])

    def ajouter(self, session_id: str, *messages: dict) -> None:
        with self._lock:
            _, file = self._data.pop(session_id, (0, deque(maxlen=self.max_messages)))
            file.extend(messages)
            self._data[session_id] = (time.monotonic(), file)
            while len(self._data) > self.max_sessions:
                self._data.popitem(last=False)

    def effacer(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)


historiques = HistoriqueSessions()
SESSION_ID_VALIDE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def _session_id(payload: dict) -> str:
    sid = str(payload.get("session_id", ""))
    return sid if SESSION_ID_VALIDE.match(sid) else str(uuid.uuid4())

# Limite globale d'appels LLM simultanés
_slots_llm = threading.BoundedSemaphore(4)


# ==============================================================================
# 4. ROUTES
# ==============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats", methods=["GET"])
def api_stats():
    # Agrégats Supabase (fonction rimel_stats) : aucun montant ni donnée personnelle.
    return jsonify(stats_inscriptions.statistiques())


@app.route("/data/geojson/<path:nom_fichier>")
def geojson(nom_fichier):
    # Seul le dossier data/geojson est exposé (la base vectorielle et les JSON des agents restent privés).
    return send_from_directory(
        os.path.join(BASE_DIR, "data", "geojson"), nom_fichier, mimetype="application/geo+json"
    )


@app.route("/api/chat", methods=["POST"])
@limiter.limit("6 per minute")
def api_chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    session_id = _session_id(payload)

    if not message:
        return jsonify({"reply": "Veuillez poser une question.", "session_id": session_id}), 400
    if len(message) > 500:
        return jsonify({"reply": "Votre question est trop longue (500 caractères maximum).", "session_id": session_id}), 400

    historique = historiques.lire(session_id)

    def repondre(texte: str, source: str, theme: str | None = None):
        historiques.ajouter(
            session_id,
            {"role": "user", "content": message},
            {"role": "assistant", "content": texte, "theme": theme},
        )
        return jsonify({"reply": texte, "source": source, "theme": theme, "session_id": session_id})

    # 1) FAQ locale instantanée (0 coût LLM)
    faq = reponse_faq(message)
    if faq:
        return repondre(faq, "faq", theme_indicatif(message))

    # 2) Cache : seulement en début de conversation (une relance dépend du contexte)
    cle = _normaliser(message)
    if not historique:
        en_cache = cache_reponses.get(cle)
        if en_cache:
            return repondre(en_cache, "cache", theme_indicatif(message))

    # 3) Orchestration multi-agents, plafonnée en concurrence
    if not _slots_llm.acquire(timeout=10):
        return jsonify({"reply": message_surcharge(), "session_id": session_id}), 503
    try:
        theme_suggere = str(payload.get("theme", "")) or None
        reponse, theme = orchestrer(message, historique, theme_suggere)
    except Exception:
        app.logger.exception("Erreur inattendue dans /api/chat")
        return jsonify({"reply": "Désolé, une erreur technique est survenue. Merci de réessayer.",
                        "session_id": session_id}), 500
    finally:
        _slots_llm.release()

    if isinstance(reponse, ReponseIndisponible):
        # Réponse de repli : affichée mais ni mémorisée ni mise en cache.
        return jsonify({"reply": reponse, "source": "indisponible", "session_id": session_id})
    if not historique:
        cache_reponses.set(cle, reponse)
    return repondre(reponse, "llm", theme)


@app.route("/api/chat/reset", methods=["POST"])
@limiter.limit("20 per minute")
def api_chat_reset():
    payload = request.get_json(silent=True) or {}
    sid = str(payload.get("session_id", ""))
    if SESSION_ID_VALIDE.match(sid):
        historiques.effacer(sid)
    return jsonify({"ok": True})


def _prechauffer_rag() -> None:
    """Charge le modèle d'embeddings et ChromaDB en arrière-plan (sinon ~1 min sur la 1re question patrimoine)."""
    try:
        import rag_service
        rag_service.rechercher_contexte("préchauffage", n_results=1)
        app.logger.info("Base RAG préchauffée.")
    except Exception:
        app.logger.exception("Préchauffage RAG impossible")


threading.Thread(target=_prechauffer_rag, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
