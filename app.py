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
    "urgence": "En cas d'urgence sur le parcours ou dans la forêt de Rimel, contactez immédiatement la Protection Civile au 198 ou adressez-vous aux postes de secours CAP Ambulance (équipements catégorie A).",
    "198": "Le 198 est le numéro d'urgence officiel de la Protection Civile en Tunisie.",

    # Écologie, Mascotte & Valeurs
    "chardonneret": "Le chardonneret (طير الحسون) est la mascotte officielle et le symbole de cette édition. Il incarne la liberté et sensibilise à la protection des 74 espèces d'oiseaux menacées en Tunisie ainsi qu'à la préservation de la biodiversité de la forêt de Rimel.",
    "oiseau": "L'édition met à l'honneur le chardonneret (طير الحسون) pour sensibiliser à la protection des 74 espèces d'oiseaux menacées en Tunisie et lutter contre leur capture illégale.",
    "ecologie": "L'événement s'inscrit dans la démarche Net Zero Trail : zéro gobelet jetable, préservation de la forêt de Rimel et actions de reboisement en partenariat avec la Direction Générale des Forêts, Ecopact et le WWF North Africa.",
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
    "menzel": "Le Rimel Challenge bénéficie d'un ancrage local fort, organisé avec le soutien précieux de la Municipalité de Menzel Jemil et de l'association We Run Rimel.",
    "we run rimel": "L'association locale We Run Rimel et la Municipalité de Menzel Jemil collaborent étroitement à l'organisation et au balisage du Rimel Challenge.",
    "heberg": "L'organisation propose une formule officielle Hébergement + petit-déjeuner à 30 DT (réservable sur Sportsense.tn). Des hôtels sont également disponibles sur la corniche de Bizerte (~10 min).",
    "navette": "Une navette officielle aller-retour Tunis / Village du Trail est proposée à 10 DT (option sélectionnable lors de l'inscription sur Sportsense.tn).",
    "horaire": "Horaires officiels : 06h30 ouverture du village, 07h00 départ du 50 km, 08h15 briefing, 08h30 départ groupé 10 km et 22 km, 08h45 départ de la Course des Super Héros Kids (1 km)."
}


def reponse_faq(question: str) -> str | None:
    q = _normaliser(question)
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
