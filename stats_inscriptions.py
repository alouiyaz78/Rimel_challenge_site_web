"""Statistiques d'inscription en direct du Rimel Challenge (Supabase).

Lecture REST de la table `registrations` (colonnes neutres uniquement : distance, paiement, nat, sexe,
dossard), agrégation en mémoire côté serveur : le navigateur ne reçoit que des totaux, jamais une ligne
individuelle ni un montant.

Règle de statut (la table n'a pas de colonne de statut de paiement) :
- validé      : paiement par carte bancaire (réglé en ligne) OU dossard attribué ;
- préinscrit  : paiement en espèces sans dossard (en attente de règlement en bureau ou le Jour J).
"""

import json
import logging
import os
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FICHIER_INSTANTANE = os.path.join(BASE_DIR, "data", "stats_instantane.json")
DUREE_CACHE = 25  # secondes : le site interroge /api/stats toutes les 30 s par visiteur
TAILLE_PAGE = 1000  # maximum renvoyé par Supabase par requête
FILTRE_RIMEL = "ilike.*Rimel Challenge By Natilait*"

COURSES = [  # (clé, libellé, motif recherché dans le libellé "distance")
    ("50K", "Ultra-Trail 50 km", "50km"),
    ("22K", "Trail 22 km", "22km"),
    ("10K", "Trail Découverte 10 km", "decouverte"),
    ("MINI", "Mini-Trail 10 km", "mini"),
    ("KIDS", "Trail Kids 1 km", "kids"),
]

PAYS = {
    "TN": "Tunisie", "FR": "France", "DZ": "Algérie", "MA": "Maroc", "LY": "Libye", "IT": "Italie",
    "ES": "Espagne", "DE": "Allemagne", "BE": "Belgique", "CH": "Suisse", "GB": "Royaume-Uni",
    "US": "États-Unis", "CA": "Canada", "NO": "Norvège", "IN": "Inde", "CN": "Chine", "CM": "Cameroun",
    "BJ": "Bénin", "SN": "Sénégal", "CI": "Côte d'Ivoire", "EG": "Égypte", "NL": "Pays-Bas", "PT": "Portugal",
}

_verrou = threading.Lock()
_cache = {"horodatage": 0.0, "donnees": None}


def _normaliser(texte) -> str:
    texte = unicodedata.normalize("NFKD", str(texte or "").lower())
    return "".join(c for c in texte if not unicodedata.combining(c)).replace(" ", "")


def _course(distance: str) -> str | None:
    d = _normaliser(distance)
    return next((cle for cle, _, motif in COURSES if motif in d), None)


def calculer(lignes: list[dict]) -> dict:
    """Agrège des lignes groupées {distance, paiement, dossard_attribue, nat, sexe, n}."""
    courses = {cle: {"code": cle, "nom": nom, "total": 0, "valides": 0, "preinscrits": 0} for cle, nom, _ in COURSES}
    demo = {"hommes": 0, "femmes": 0, "enfants": 0, "non_renseigne": 0}
    pays: dict[str, int] = {}
    total = valides = 0

    for l in lignes:
        n = int(l.get("n") or 0)
        cle = _course(l.get("distance"))
        if not cle or n <= 0:
            continue
        valide = bool(l.get("dossard_attribue")) or "carte" in _normaliser(l.get("paiement"))
        total += n
        valides += n if valide else 0
        c = courses[cle]
        c["total"] += n
        c["valides" if valide else "preinscrits"] += n

        sexe = _normaliser(l.get("sexe"))
        if cle == "KIDS":
            demo["enfants"] += n
        elif sexe.startswith(("homme", "h", "m")):
            demo["hommes"] += n
        elif sexe.startswith(("femme", "f")):
            demo["femmes"] += n
        else:
            demo["non_renseigne"] += n

        code = str(l.get("nat") or "").strip().upper()
        if code:
            pays[code] = pays.get(code, 0) + n

    liste_pays = sorted(
        ({"code": c, "nom": PAYS.get(c, c), "inscrits": n} for c, n in pays.items()),
        key=lambda p: -p["inscrits"],
    )
    return {
        "disponible": True,
        "total": total,
        "valides": valides,
        "preinscrits": total - valides,
        "nb_pays": len(liste_pays),
        "demographie": demo,
        "courses": list(courses.values()),
        "pays": liste_pays,
        "mis_a_jour": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _lire_registrations() -> list[dict]:
    """Lit les inscriptions Rimel (colonnes neutres), page par page, et les regroupe en comptages."""
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    cle = os.environ.get("SUPABASE_KEY", "")
    if not url or not cle:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY absents")
    if not url.endswith("/rest/v1"):  # accepte l'URL du projet comme celle de l'API REST
        url += "/rest/v1"
    requete = (f"{url}/registrations?select=distance,paiement,nat,sexe,dossard"
               f"&distance={urllib.parse.quote(FILTRE_RIMEL)}&order=id")

    groupes: Counter = Counter()
    debut = 0
    while True:
        entetes = {"apikey": cle, "Authorization": f"Bearer {cle}", "Accept": "application/json",
                   "Range-Unit": "items", "Range": f"{debut}-{debut + TAILLE_PAGE - 1}"}
        with urllib.request.urlopen(urllib.request.Request(requete, headers=entetes), timeout=10) as reponse:
            if reponse.status not in (200, 206):
                raise RuntimeError(f"HTTP {reponse.status}")
            lot = json.load(reponse)
        for l in lot:
            groupes[(l.get("distance"), l.get("paiement"), bool(str(l.get("dossard") or "").strip()),
                     l.get("nat"), l.get("sexe"))] += 1
        if len(lot) < TAILLE_PAGE:
            break
        debut += TAILLE_PAGE

    return [{"distance": d, "paiement": p, "dossard_attribue": a, "nat": n, "sexe": sx, "n": nb}
            for (d, p, a, n, sx), nb in groupes.items()]


def _instantane() -> dict | None:
    try:
        with open(FICHIER_INSTANTANE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def statistiques() -> dict:
    """Stats en direct (cache 25 s) ; repli sur le dernier instantané si Supabase est injoignable."""
    with _verrou:
        if _cache["donnees"] and time.monotonic() - _cache["horodatage"] < DUREE_CACHE:
            return _cache["donnees"]
        try:
            donnees = {**calculer(_lire_registrations()), "source": "supabase"}
            try:
                with open(FICHIER_INSTANTANE, "w", encoding="utf-8") as f:
                    json.dump(donnees, f, ensure_ascii=False)
            except OSError:
                logger.warning("Instantané des stats non enregistré")
        except Exception as e:
            logger.warning("Supabase injoignable (%s) : repli sur l'instantané local", e)
            instantane = _instantane()
            donnees = {**instantane, "source": "instantane"} if instantane else {"disponible": False, "source": "aucune"}
        _cache.update(horodatage=time.monotonic(), donnees=donnees)
        return donnees
