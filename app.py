import json
import os
import re
import threading
import time
import unicodedata
from collections import OrderedDict

from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from agents import BASE_DIR, ReponseIndisponible, message_surcharge, traiter_question_visiteur

app = Flask(__name__)

# ==============================================================================
# 1. RATE LIMITING PAR IP (Flask-Limiter)
# ==============================================================================
limiter = Limiter(
    key_func=get_remote_address,
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
    "urgence": "En cas d'urgence sur le parcours ou dans la forêt de Rimel, contactez immédiatement la protection civile au 198 ou adressez-vous aux postes de secours CAP Ambulance.",
    "198": "Le 198 est le numéro officiel de la protection civile pour toute urgence sur le parcours.",

    # Écologie, Mascotte & Valeurs
    "gobelet": "Le gobelet réutilisable personnel est OBLIGATOIRE ♻️. Aucun gobelet jetable n'est fourni sur les ravitaillements (démarche Net Zero Trail).",
    "jetable": "Les gobelets jetables sont strictement interdits dans le cadre du Net Zero Trail. Vous devez apporter votre gobelet réutilisable personnel.",
    "chardonneret": "Le chardonneret (طير الحسون) est le symbole de cette édition pour sensibiliser à la protection des 74 espèces d'oiseaux menacées en Tunisie et préserver la biodiversité de la forêt de Rimel.",
    "oiseau": "L'édition met à l'honneur le chardonneret sauvage pour sensibiliser à la lutte contre la capture illégale et préserver les espèces forestières.",
    "ecologie": "L'événement s'inscrit dans la démarche Net Zero Trail : zéro plastique jetable, reboisement des forêts avec le WWF North Africa et respect strict du milieu naturel.",

    # Courses & Points ITRA
    "kids": "La course Kids (1 km), baptisée 'Course des Super Héros', est ouverte à tous les enfants (10 DT) pour découvrir le trail en toute sécurité.",
    "itra": "Points ITRA officiels attribués : 1 point ITRA pour le Trail 22 km et 2 points ITRA pour le grand défi de l'Ultra-Trail 50 km.",
    "50 km": "L'Ultra-Trail 50 km est le grand défi de l'édition : 100 DT (dossard + t-shirt officiel inclus), accorde 2 points ITRA. Départ à 07h00.",
    "22 km": "Le Trail 22 km (35 DT) est chronométré avec puce électronique et rapporte 1 point ITRA. Départ à 08h30.",
    "10 km": "Deux formats 10 km sont proposés : Trail Découverte sans puce (15 DT) et Mini-Trail chronométré avec puce (20 DT). Départ à 08h30.",

    # Logistique, Tarifs & Localisation
    "menzel": "Le Rimel Challenge est organisé en étroite collaboration avec la municipalité de Menzel Jemil et l'association We Run Rimel.",
    "heberg": "L'organisation propose une formule officielle Hébergement + petit-déjeuner à 30 DT (réservable sur Sportsense.tn). Des hôtels sont aussi disponibles sur la corniche de Bizerte.",
    "navette": "Une navette officielle aller-retour Tunis / Village du Trail est disponible à 10 DT lors de l'inscription.",
    "horaire": "Programme : 06h30 ouverture, 07h00 départ du 50 km, 08h15 briefing, 08h30 départ groupé 10 km et 22 km, 08h45 Course des Super Héros Kids (1 km)."
}


def reponse_faq(question: str) -> str | None:
    q = _normaliser(question)
    for keywords, reponse in REPONSES_COMMUNES.items():
        if any(kw in q for kw in keywords):
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

# Limite globale d'appels LLM simultanés
_slots_llm = threading.BoundedSemaphore(4)


# ==============================================================================
# 4. ROUTES
# ==============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
@limiter.limit("6 per minute")
def api_chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()

    if not message:
        return jsonify({"reply": "Veuillez poser une question."}), 400
    if len(message) > 500:
        return jsonify({"reply": "Votre question est trop longue (500 caractères maximum)."}), 400

    # 1) FAQ locale instantanée (0 coût LLM)
    faq = reponse_faq(message)
    if faq:
        return jsonify({"reply": faq, "source": "faq"})

    # 2) Cache des questions déjà posées
    cle = _normaliser(message)
    en_cache = cache_reponses.get(cle)
    if en_cache:
        return jsonify({"reply": en_cache, "source": "cache"})

    # 3) Appel LLM, plafonné en concurrence
    if not _slots_llm.acquire(timeout=10):
        return jsonify({"reply": message_surcharge()}), 503
    try:
        reponse = traiter_question_visiteur(message)
    except Exception:
        app.logger.exception("Erreur inattendue dans /api/chat")
        return jsonify({"reply": "Désolé, une erreur technique est survenue. Merci de réessayer."}), 500
    finally:
        _slots_llm.release()

    if not isinstance(reponse, ReponseIndisponible):
        cache_reponses.set(cle, reponse)
    return jsonify({"reply": reponse, "source": "llm"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
