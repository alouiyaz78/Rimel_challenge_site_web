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
# Derrière un reverse proxy (Nginx, Render...), ajouter ProxyFix pour obtenir la vraie IP :
#   from werkzeug.middleware.proxy_fix import ProxyFix
#   app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri="memory://",  # Mono-processus ; avec plusieurs workers Gunicorn, utiliser redis://
)


@app.errorhandler(429)
def trop_de_requetes(e):
    # Le front lit `data.reply` : on garde le même format JSON
    return jsonify({
        "reply": "Doucement, coureur ! 🏃 Vous avez envoyé beaucoup de questions. "
                 "Merci de patienter une minute avant de réessayer."
    }), 429


# ==============================================================================
# 2. FAQ LOCALE (réponses instantanées, sans appel LLM)
# ==============================================================================

def _normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", texte.lower())
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", texte)).strip()


def _charger(nom_fichier: str) -> dict:
    chemin = os.path.join(BASE_DIR, "data", nom_fichier)
    if os.path.exists(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


_logistique = _charger("logistique.json")
_lien = os.environ.get("RIMEL_REGLEMENT_URL", "").strip()
_voir_reglement = f"le règlement officiel : {_lien}" if _lien else "le règlement officiel de la course"

# Clés optionnelles de data/logistique.json : renseignées par l'organisation, jamais inventées
_horaires = _logistique.get("horaires_depart") or (
    f"Les horaires officiels des départs 10K et 25K sont publiés dans {_voir_reglement}. "
    "Prévoyez d'arriver au moins 45 minutes avant le premier départ."
)
_dossards = _logistique.get("retrait_dossards") or (
    f"Les lieux et créneaux de retrait des dossards sont précisés dans {_voir_reglement}. "
    "Munissez-vous d'une pièce d'identité."
)

# Chaque entrée : (groupes de mots-clés — un mot de CHAQUE groupe doit apparaître, réponse)
FAQ = [
    (
        [("gobelet", "ecocup")],
        "Le gobelet réutilisable personnel est OBLIGATOIRE ♻️ Aucun gobelet jetable n'est fourni "
        "sur les ravitaillements. Les emballages de gels et barres doivent être conservés sur soi, "
        "sous peine de disqualification.",
    ),
    (
        [("urgence", "secours", "pompier", "protection civile", "incendie", "feu", "198")],
        "🚨 Urgence / feu de forêt : appelez la Protection Civile au 198. Il est formellement "
        "interdit de fumer, de jeter des mégots ou d'allumer du feu dans la forêt de Rimel.",
    ),
    (
        [("heure", "horaire", "quand", "commence"), ("10k", "10 km", "25k", "25 km", "course", "depart")],
        _horaires,
    ),
    (
        [("dossard",)],
        _dossards,
    ),
]


def _contient(q: str, mot: str) -> bool:
    return re.search(rf"\b{re.escape(mot)}", q) is not None


def reponse_faq(question: str) -> str | None:
    q = _normaliser(question)
    for groupes, reponse in FAQ:
        if all(any(_contient(q, mot) for mot in groupe) for groupe in groupes):
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

# Limite globale d'appels LLM simultanés, en complément de la limite par IP
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

    # 1) FAQ locale : aucun token consommé
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

    # Les messages de repli (429, erreur) ne sont jamais mis en cache
    if not isinstance(reponse, ReponseIndisponible):
        cache_reponses.set(cle, reponse)
    return jsonify({"reply": reponse, "source": "llm"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
