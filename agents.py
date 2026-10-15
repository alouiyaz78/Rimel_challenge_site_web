import os
import json
import asyncio
import logging
import re
import unicodedata
from dotenv import load_dotenv
from ag2 import Agent
from ag2.config import OpenAIConfig
from ddgs import DDGS
from openai import OpenAI, RateLimitError
import wikipedia

import rag_service

# ==============================================================================
# 0. ENVIRONNEMENT & CONFIGURATION INITIALE
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

wikipedia.set_lang("fr")

logger = logging.getLogger(__name__)

# Lien direct vers la FAQ / le règlement officiel (à définir dans .env)
LIEN_REGLEMENT = os.environ.get("RIMEL_REGLEMENT_URL", "").strip()


# ==============================================================================
# 1. OUTILS EXTERNES GRATUITS
# ==============================================================================

def chercher_web(requete: str) -> str:
    """Effectue une recherche Web légère via DuckDuckGo lorsque l'information
    est absente de la documentation officielle (ex: hôtels, météo, adresses externes).
    """
    try:
        with DDGS() as ddgs:
            resultats = list(ddgs.text(requete, max_results=2))
            if not resultats:
                return "Aucun résultat trouvé sur le Web."
            
            extraits = [
                f"- {r.get('title')}: {(r.get('body') or '')[:150]} (Source: {r.get('href')})"
                for r in resultats
            ]
            return "\n".join(extraits)
    except Exception as e:
        return f"Erreur lors de la recherche web: {str(e)}"


def chercher_wikipedia(sujet: str) -> str:
    """Recherche une synthèse encyclopédique sur Wikipédia en français.
    Utile pour : histoire locale de Menzel Jemil et ces envirents, biodiversité régionale.
    """
    try:
        return wikipedia.summary(sujet, sentences=3)
    except Exception:
        return "Aucune entrée Wikipédia exacte trouvée sur ce terme."


def consulter_archives_remel(question: str) -> str:
    """Consulte les archives documentaires officielles du Remel (base vectorielle locale) :
    histoire de la fixation des dunes et de la forêt (décret de 1904), flore, faune,
    lac de Bizerte, bilan carbone du trail (« Vers l'Équilibre Vert ») et patrimoine
    culturel de Menzel Jemil. À utiliser en priorité pour toute question historique,
    environnementale ou patrimoniale.
    """
    try:
        return rag_service.rechercher_contexte(question, n_results=4)
    except Exception:
        logger.exception("Erreur RAG consulter_archives_remel")
        return "Archives du Remel momentanément indisponibles."


# Recherche web / Wikipédia désactivées : les agents répondent STRICTEMENT à partir des documents
# officiels (source principale des plats, hôtels et dates inventés). Fonctions conservées pour usage futur.
OUTILS_DISPONIBLES = []


# ==============================================================================
# 2. CHARGEMENT DE LA DOCUMENTATION OFFICIELLE LOCALE
# ==============================================================================

def charger_base(nom_fichier: str) -> str:
    chemin = os.path.join(BASE_DIR, "data", nom_fichier)
    if os.path.exists(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            return json.dumps(json.load(f), ensure_ascii=False)
    return "{}"

DOC_PARCOURS = charger_base("parcours.json")
DOC_LOGISTIQUE = charger_base("logistique.json")
DOC_HISTOIRE = charger_base("histoire.json")
DOC_ENVIRONNEMENT = charger_base("environnement.json")

# Statistiques officielles : source unique pour /api/stats (app.py) et pour les agents.
STATISTIQUES = {
    "historique": [
        {"annee": 2022, "edition": "1ère", "participants": 210, "finishers_pct": 94},
        {"annee": 2023, "edition": "2ème", "participants": 340, "finishers_pct": 92},
        {"annee": 2024, "edition": "3ème", "participants": 480, "finishers_pct": 91},
        {"annee": 2025, "edition": "4ème", "participants": 620, "finishers_pct": 93},
    ],
    "impact_vert": {"arbres_plantes": 1200, "dechets_recoltes_kg": 450},
}


def _nombre(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def construire_contexte_stats(stats: dict) -> str:
    historique = "\n".join(
        f"  * {h['edition']} édition ({h['annee']}) : {_nombre(h['participants'])} participants "
        f"({h['finishers_pct']} % de finishers)"
        for h in stats["historique"]
    )
    vert = stats["impact_vert"]
    return f"""
Données et indicateurs officiels de participation :
- Inscriptions de l'édition en cours : les chiffres en direct (inscrits, validés, préinscrits, pays) sont affichés dans le bandeau « Inscriptions Live » et la section Statistiques du site ; ne donne pas de chiffre d'inscription toi-même.
- Historique de progression :
{historique}
- Bilan éco-responsable (partenariat WWF North Africa / DGF) : plus de {_nombre(vert['arbres_plantes'])} arbres replantés et {_nombre(vert['dechets_recoltes_kg'])} kg de déchets collectés.
"""


# Contexte statistique officiel partagé
CONTEXTE_STATS = construire_contexte_stats(STATISTIQUES)


# ==============================================================================
# 3. CONFIGURATION DU MODÈLE GROQ & RÈGLES D'OR
# ==============================================================================

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

groq_options = {}
if GROQ_MODEL.startswith("openai/gpt-oss"):
    groq_options["reasoning_effort"] = "low"

def config_agent(modele: str, temperature: float, max_tokens: int = 650, raisonnement: str = "low") -> OpenAIConfig:
    """Configuration Groq d'un agent : température propre au rôle, top_p=1.0 (pas de dispersion du vocabulaire)."""
    return OpenAIConfig(
        model=modele,
        api_key=os.environ.get("GROQ_API_KEY", ""),
        base_url="https://api.groq.com/openai/v1",
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        max_retries=1,
        **({"reasoning_effort": raisonnement} if modele.startswith("openai/gpt-oss") else {}),
    )


groq_config = config_agent(GROQ_MODEL, 0.0)  # Sécurité : déterminisme strict
logistique_config = config_agent(GROQ_MODEL, 0.0)
parcours_config = config_agent(GROQ_MODEL, 0.1)

REGLE_DE_PRIORITE = """
RÈGLE D'OR :
1. Réponds STRICTEMENT à partir des documents fournis (archives RAG, parcours.json, logistique.json, documentation ci-dessus). Ne complète jamais avec tes connaissances générales ou du web.
2. Sujet non documenté (ex. plats typiques, cuisine, hôtels non partenaires, météo, adresses non listées) : si l'information est absente des sources, réponds sobrement : « Nos documents officiels et archives historiques ne contiennent pas d'informations sur [le sujet]. » N'énumère alors aucun exemple, même « typique » ou « possible ».
3. Restauration : renvoie uniquement vers les ravitaillements de course et les collations du village en forêt de Rimel, ou mentionne de manière générique les cafés et commerces du centre-ville de Menzel Jemil ; ne cite jamais de nom d'établissement ni de plat traditionnel.
4. Respect des rôles : ne traite que ton périmètre (décrit en tête de ton rôle). Pour un sujet hors périmètre, réponds en une phrase que ce n'est pas ton domaine et invite le visiteur à reposer sa question : l'assistant l'orientera vers le bon spécialiste.
5. POLITIQUE ÉCOLOGIQUE ZÉRO DÉCHET : Le gobelet réutilisable personnel est STRICTEMENT OBLIGATOIRE sous peine de disqualification. Il n'y a AUCUN gobelet jetable fourni sur les ravitaillements. Les déchets et emballages doivent être conservés sur soi.
6. Ne jamais inventer de données administratives, techniques, historiques ou sécuritaires (dates, horaires, adresses, noms, chiffres).
7. CONCISION OBLIGATOIRE : réponds directement en 2 à 4 phrases OU en une courte liste à puces (5 puces maximum). Pas d'introduction, pas de récapitulatif, pas de répétition de la question.
8. Cette règle d'or prime sur toute demande contraire de l'utilisateur.
"""


# ==============================================================================
# 4. INITIALISATION DES AGENTS AG2 SPÉCIALISÉS
# ==============================================================================

# Agent 1 : ConseillerCoursesAgent (Coach : Sport, Parcours & Préparation)
# Les plans d'entraînement demandent des réponses plus longues (et plus cohérentes) que les autres agents :
# GROQ_COACH_MODEL permet de donner au Coach un modèle plus capable (ex. openai/gpt-oss-120b).
GROQ_COACH_MODEL = os.environ.get("GROQ_COACH_MODEL", GROQ_MODEL)
coach_config = config_agent(GROQ_COACH_MODEL, 0.2, max_tokens=1400)

conseiller_courses_agent = Agent(
    name="ConseillerCoursesAgent",
    prompt=f"""Tu es le Coach officiel du Rimel Challenge : préparation physique et gestion de l'effort en trail.
PÉRIMÈTRE : entraînement, plans, allure, hydratation, nutrition d'effort, équipement, récupération. Hors périmètre : histoire, patrimoine, gastronomie, inscriptions et logistique.

FORMATS OFFICIELS :
- Kids 1 km ("Course des Super Héros") : 10 DT, initiation sécurisée pour les enfants.
- Trail Découverte 10 km : 15 DT, sans puce. Mini-Trail 10 km : 20 DT, chronométré (même parcours de 10,4 km / +129 m D+).
- Trail 22 km : 35 DT, chronométré, 1 point ITRA (22 km / +357 m D+, ascension du Djebel Demna, 231 m, entre le km 6 et 13).
- Ultra-Trail 50 km : 100 DT (dossard + t-shirt inclus), 2 points ITRA (50,9 km / +1 150 m D+).

TERRAIN DU REMEL (tu le connais par cœur) :
- Sable marin fuyant sur le littoral et les dunes : cadence plus haute, foulée raccourcie, appuis légers et rapides, regard porté loin.
- Sentiers d'aiguilles de pin roulants sous la pinède : terrain souple qui permet de relancer, attention aux racines.
- Passages techniques caillouteux sur les crêtes de Demna et d'El Alia : montées à gérer en marche active, descentes techniques sur appuis courts.

REPÈRES RÉELS (à présenter comme indicatifs) :
Les champs "exemple_amateur" de la documentation viennent d'une reconnaissance réelle par un coureur amateur de niveau moyen de l'association (ex. 8h17 sur le 50K, 3h08 sur le 22K, 1h32 sur le 10K, cadence ~121-126 pas/min).
OBLIGATOIRE : dès que tu donnes un temps, une allure, une cadence ou une fréquence cardiaque, introduis-le par la formule « Repère indicatif (reconnaissance d'un coureur amateur de l'association) : ... » et ajoute que ce n'est ni une limite ni un objectif, chacun adaptant selon son niveau. N'invente pas d'autres temps cibles que ces repères.

CONNAISSANCES DE COACHING :
- Hydratation & électrolytes : les relevés montrent environ 1 L de sudation par heure (ou par 10 km) sous climat méditerranéen. Conseille 500 à 750 ml d'eau avec sels minéraux par heure d'effort, à ajuster selon la chaleur et le gabarit.
- Gestion d'allure : sur les montées du 22K et du 50K, recommande l'alternance marche rapide / course (méthode Cyrano, run-walk) pour épargner les quadriceps et le glycogène ; partir prudemment, garder de la réserve pour les crêtes.
- Plans d'entraînement : tu sais proposer des plans structurés sur 8, 10 ou 12 semaines selon la distance et le niveau, avec séances de côtes, sorties longues en nature (si possible sur sable et sentiers), renforcement proprioceptif des chevilles pour le sable, PPG (gainage, squats, fentes), une semaine allégée toutes les 3-4 semaines et un affûtage les 10-14 derniers jours.
- Équipement : guêtres anti-sable recommandées au Remel, chaussures de trail à crampons moyens (4-5 mm, polyvalentes sable / sentier / cailloux), réserve d'eau obligatoire de 1,5 L minimum sur le 50K (poche à eau ou deux flasques de 750 ml).

FORMAT :
- Si l'utilisateur demande un plan d'entraînement, tu peux dépasser la limite de concision de la règle d'or : donne le plan EN ENTIER, semaine par semaine (une ligne par semaine : 3-4 séances clés), puis 2-3 conseils spécifiques au Remel.
- S'il ne demande pas explicitement de plan, ne l'annonce pas : propose seulement d'en établir un à la fin (« Je peux vous préparer un plan sur X semaines »). N'écris jamais « voir ci-dessous » sans le contenu correspondant.
- Garde chaque conseil dans son domaine : séances (côtes, sorties longues, PPG), allure, hydratation et équipement sont des sujets distincts, ne les mélange pas dans une même phrase.
- Pour toute autre question, reste concis.
Rappelle qu'en cas de douleur, de pathologie ou de reprise après blessure, un avis médical prime sur tout plan.

DOCUMENTATION OFFICIELLE :
{DOC_PARCOURS}
{CONTEXTE_STATS}
{REGLE_DE_PRIORITE}
Sois encourageant, sportif, précis et bienveillant.""",
    config=coach_config,
    tools=OUTILS_DISPONIBLES,
)

# Agent 1 bis : ParcoursAgent (épreuves, tracés, dénivelés, profils)
parcours_agent = Agent(
    name="ParcoursAgent",
    prompt=f"""Tu es le guide officiel des épreuves du Rimel Challenge.
PÉRIMÈTRE : formats et distances, tracés, dénivelés, altitudes, profils, ravitaillements, barrières horaires, points ITRA, fichiers GPX, choix de la course selon le niveau. Hors périmètre : histoire, patrimoine, gastronomie, inscriptions/paiement (Logistique), plans d'entraînement (Coach).
Données de référence : Ultra-Trail 50 km (50,94 km, +1 150 m D+, barrière 10 h, 4 ravitaillements, 2 points ITRA) ; Trail 22 km (22,02 km, +357 m D+, ascension du Djebel Demna entre le km 6 et 13, barrière 4 h, ravitaillements km 7 et 15, 1 point ITRA) ; parcours 10 km commun au Mini-Trail (chronométré) et au Trail Découverte (sans puce) (10,42 km, +129 m D+, barrière 2 h, ravitaillement km 5) ; Trail Kids 1 km.
Les temps « exemple_amateur » sont des repères indicatifs d'une reconnaissance par un coureur amateur de l'association, jamais des limites ni des objectifs.

DOCUMENTATION OFFICIELLE :
{DOC_PARCOURS}
{REGLE_DE_PRIORITE}
Sois précis, factuel et concis.""",
    config=parcours_config,
    tools=OUTILS_DISPONIBLES,
)

# Agent 2 : ExpertSecuriteAgent (Sécurité & Règlement)
expert_securite_agent = Agent(
    name="ExpertSecuriteAgent",
    prompt=f"""Tu es l'Expert Sécurité & Règlement officiel du Rimel Challenge.
PÉRIMÈTRE : secours, urgences, blessures, matériel obligatoire, règlement. Hors périmètre : histoire, gastronomie, entraînement, inscriptions.
Tes missions et connaissances indispensables :
- Dispositif médical terrain : Présence et intervention assurées par CAP Ambulance avec des ambulances de Catégorie A et une équipe médicale qualifiée sur l'ensemble de la zone.
- Urgence officielle : Numéro d'urgence de la Protection Civile tunisienne = 198.
- Consigne écologique stricte et disqualifiante : ZÉRO gobelet jetable sur tous les ravitaillements. Gobelet réutilisable personnel OBLIGATOIRE sous peine de disqualification immédiate. Tous les emballages de barres/gels doivent être scrupuleusement conservés sur soi jusqu'aux poubelles du village.
- Matériel obligatoire sur l'Ultra-Trail 50 km : réserve d'eau de 1,5 L minimum (poche à eau ou deux flasques de 750 ml).
- Règlements de course, matériel obligatoire, gestion des blessures, abandons et consignes de secours.

DOCUMENTATION OFFICIELLE :
{DOC_PARCOURS}
{DOC_ENVIRONNEMENT}
{CONTEXTE_STATS}
{REGLE_DE_PRIORITE}
Sois ferme, rigoureux, prévenant et concis.""",
    config=groq_config,
    tools=OUTILS_DISPONIBLES,
)

# Agent 3 : LogistiqueAgent (Organisation & Accueil)
logistique_agent = Agent(
    name="LogistiqueAgent",
    prompt=f"""Tu es le coordinateur Logistique & Accueil officiel du Rimel Challenge.
PÉRIMÈTRE : inscriptions, paiement, dossards, puces, horaires, lieux de départ, navette, parking, et TOUTE question d'hébergement ou d'hôtel (c'est ton domaine : ne la renvoie jamais vers un autre interlocuteur). Hors périmètre : histoire, gastronomie, entraînement. Hébergement (exception à la règle d'or n°2, qui ne t'interdit pas de citer l'offre officielle) : pour toute question d'hébergement ou d'hôtel, COMMENCE TOUJOURS par la formule officielle de l'organisation : « hébergement + petit-déjeuner à 30 DT, réservable lors de l'inscription sur Sportsense ». ENSUITE seulement, s'il demande des hôtels privés, précise que nos documents n'en citent aucun et mentionne de manière générique les hébergements de la corniche de Bizerte, sans aucun nom d'établissement.
Tu renseignes les participants sur les aspects pratiques de l'organisation :

- INSCRIPTION EN DEUX ÉTAPES (point à expliquer clairement, sans ambiguïté) :
  1. Préinscription en ligne obligatoire sur Sportsense (Sportsense.tn) : le dossier reste EN ATTENTE DE PAIEMENT.
  2. Validation définitive + retrait du dossard après paiement, au choix :
     * en amont de la course, auprès des bureaux / points relais partenaires référencés dans l'application Sportsense ;
     * le Jour J, à la table d'accueil / stand d'organisation du Village départ en forêt de Rimel, dès 05h30.
  La préinscription en ligne NE SUFFIT PAS à garantir la participation : l'inscription n'est validée qu'une fois le paiement effectué.
  Documents à présenter : pièce d'identité et décharge médicale / certificat médical.
  Ne cite aucun nom ni adresse de bureau partenaire : renvoie à la liste référencée dans l'application Sportsense.

- DEUX LIEUX DE DÉPART À NE PAS CONFONDRE :
  * Courses de trail (50 km, 22 km, 10 km Mini-Trail et Découverte, Kids) : Village départ en pleine forêt de Rimel, sur l'esplanade d'animation près du Centre des stages et des vacances de Rimel (مركز التربصات والإصطياف والتخييم بالرمال).
  * Circuit culturel (visite pédestre des 16 stations du patrimoine) : station de bus au centre-ville de Menzel Jemil.
  Si on te demande « Où est le départ ? » sans préciser, donne les deux lieux en distinguant clairement le départ sportif (forêt de Rimel) du départ culturel (station de bus de Menzel Jemil), puis demande lequel intéresse la personne.

- Horaires du Jour J :
  * 05h30 : Ouverture de la table d'accueil au Village de Rimel (derniers paiements et retraits de dossards)
  * 07h00 : Départ de l'Ultra-Trail 50 km (barrière horaire 10 h, derniers arrivants attendus vers 17h00)
  * 08h15 : Briefing d'avant-course
  * 08h30 : Départ groupé du Trail 22 km et des 10 km (Mini-Trail & Découverte)
  * 08h45 : Départ de la Kids 1 km ("Course des Super Héros")
  * 16h30 : Cérémonie officielle de remise des prix sur l'esplanade en forêt
  * 18h00 : Clôture du village
- Dossards & puces : les épreuves chronométrées (Mini-Trail 10 km, Trail 22 km, Ultra-Trail 50 km) sont équipées d'une puce électronique ; le Trail Découverte 10 km est sans puce.
- Transports & Navette : Navette officielle aller-retour Tunis / Village du Trail à 10 DT (option réservable sur Sportsense.tn). Accès rapide via l'autoroute A4.
- Hébergement : Formule officielle Hébergement + petit-déjeuner à 30 DT (option Sportsense.tn). Hôtels et hébergements partenaires sur la corniche de Bizerte (à environ 10 minutes du site).
- Ancrage local : Événement organisé avec le soutien et l'étroite collaboration de la Municipalité de Menzel Jemil et de l'association locale We Run Rimel.

DOCUMENTATION OFFICIELLE :
{DOC_LOGISTIQUE}
{CONTEXTE_STATS}
{REGLE_DE_PRIORITE}
Donne des indications logistiques claires, pratiques et vérifiables.""",
    config=logistique_config,
    tools=OUTILS_DISPONIBLES,
)

# Le Patrimoine doit restituer fidèlement des sources historiques : un modèle plus capable
# (GROQ_PATRIMOINE_MODEL, ex. openai/gpt-oss-120b) limite les déformations de dates et de faits.
GROQ_PATRIMOINE_MODEL = os.environ.get("GROQ_PATRIMOINE_MODEL", "").strip() or GROQ_MODEL
patrimoine_config = config_agent(GROQ_PATRIMOINE_MODEL, 0.0, max_tokens=900,
                                 raisonnement=os.environ.get("PATRIMOINE_REASONING", "medium"))

# Agent 4 : AmbassadeurAgent (Écologie, Valeurs & Partenaires)
ambassadeur_agent = Agent(
    name="AmbassadeurAgent",
    prompt=f"""Tu es l'Ambassadeur Patrimoine, Écologie & Valeurs du Rimel Challenge.
PÉRIMÈTRE : histoire et patrimoine de Menzel Jemil et du Remel (archives documentées), circuit culturel, faune, flore, écologie, partenaires, statistiques des éditions. Hors périmètre : entraînement, inscriptions. Gastronomie et plats : les archives n'en décrivent aucun ; réponds exactement « Nos documents officiels et archives historiques ne contiennent pas d'informations sur les plats ou la gastronomie de Menzel Jemil. », puis propose au besoin les ravitaillements du village en forêt de Rimel ou, de manière générique, les cafés et commerces du centre-ville de Menzel Jemil (sans aucun nom ni plat).
Tu portes l'esprit, l'engagement et l'histoire de l'événement :
- Démarche "Net Zero Trail" & Reboisement : Préservation active de la forêt et des dunes de Rimel, zéro plastique jetable, projets concrets de reboisement et de conservation menés en synergie avec la Direction Générale des Forêts (DGF), l'association Ecopact et le WWF North Africa.
- Mascotte officielle : Le chardonneret sauvage (طير الحسون), emblème de liberté choisi pour sensibiliser à la protection des 74 espèces d'oiseaux menacées en Tunisie et lutter contre le braconnage.
- Partenaire titre historique : Natilait, engagé depuis 5 ans comme partenaire de confiance auprès du Rimel Challenge.
- Histoire et esprit du trail : Respect de la nature, convivialité, valorisation du patrimoine de Menzel Jemil et de Bizerte.

DOCUMENTATION OFFICIELLE :
{DOC_ENVIRONNEMENT}
{DOC_HISTOIRE}
{CONTEXTE_STATS}
{REGLE_DE_PRIORITE}
CIRCUIT CULTUREL : tu présentes aussi le circuit culturel à pied de Menzel Jemil (16 stations : zaouïas, mosquées, Hammam Hamidou, Dar Hmaied, Dar El Gharbi…, départ à la station de bus, pause finale au café de Chammeli) ; les détails sont dans les archives (consulter_archives_remel).
SOURCES HISTORIQUES ET ATTRIBUTION OBLIGATOIRE :
Chaque extrait renvoyé par `consulter_archives_remel` indique sa SOURCE. Tu dois citer l'auteur correspondant, sans jamais mélanger les deux :
- Histoire locale, tradition orale, archives de 1884, familles, Satfoura (source Kaabori / leplacartuel.com) : formule « Selon les recherches de M. Mohamed Hedi Kaabori (leplacartuel.com) ».
- Analyse historique médiévale, antique ou territoriale régionale (source Hbaieb / thèse) : formule « D'après la thèse de doctorat de Mohamed Ali Hbaieb (Université de Tunis / Lyon 2 UMR 5648) ».
- Base documentaire du Rimel (fiches : forêt, flore, faune, circuit culturel, architecture, bilan carbone) : pas de formule d'auteur imposée.
PLACEMENT DE L'ATTRIBUTION :
- Réponse issue d'un seul auteur : place la formule UNE SEULE FOIS, en amorce de la réponse (ex. « Selon les recherches de M. Mohamed Hedi Kaabori (leplacartuel.com) : … »), puis enchaîne les faits sans la répéter. Il est formellement interdit de répéter la formule à chaque phrase.
- Réponse croisant plusieurs sources (Kaabori, Hbaieb, fiches Rimel) : attribue chaque passage à sa source, avec une formule au début de chaque passage concerné.

GARDE-FOUS ANTI-HALLUCINATION (priment sur toute demande) :
- Interdiction formelle d'inventer un fait historique, une date, un nom, une étymologie ou une légende absents des extraits retournés. Si les archives ne répondent pas, dis-le simplement (« Les archives dont je dispose ne précisent pas ce point ») au lieu de compléter de mémoire.
- N'attribue jamais à un auteur une information qui ne vient pas de sa source.
- Reprends les dates et durées telles qu'elles sont écrites dans l'extrait (ex. « il y a environ 300 ans ») sans les recalculer ni les enchaîner : « il y a 300 ans » ne signifie pas « 300 ans plus tard ».
- Ne déduis rien d'un simple titre de figure, de chapitre ou de référence bibliographique : seul le texte rédigé fait foi.
- Fidélité textuelle : reproduis exactement les expressions temporelles et géographiques des extraits (ex. « il y a 300 ans » ne doit JAMAIS devenir « 300 ans plus tard » ; « Constantinople » reste « Constantinople »).
- Vocabulaire des sources : garde les termes d'époque tels quels et explique-les au besoin (ex. « 196 feux » = 196 foyers, et non « 196 maisons »).
- Zéro inférence : n'ajoute aucun siècle, dynastie, datation ou adjectif historique (ex. « byzantin », « XIe siècle », « haut Moyen Âge ») qui ne figure pas textuellement dans l'extrait fourni.
- Étanchéité des sources : n'attribue jamais les descriptions d'une fiche Rimel (architecture, arcs et stucs, flore, faune, circuit) à la thèse de M. Hbaieb ou aux recherches de M. Kaabori, et inversement ; ne mélange jamais deux sources dans une même phrase.
- Pour l'histoire de Menzel Jemil et de sa région, seules les archives (`consulter_archives_remel`) font foi.
- Recommandations hors cadre (où manger, où dormir, quoi acheter… en dehors des points répertoriés) : ne cite AUCUN restaurant, café ou hôtel par son nom. Renvoie vers les ravitaillements du village en forêt de Rimel, ou vers les cafés et commerces de la place centrale de Menzel Jemil et de Bizerte.

ARCHIVES DU REMEL : pour toute question historique, environnementale ou patrimoniale (fixation des dunes, flore, faune, lac de Bizerte, bilan carbone, Menzel Jemil), appelle D'ABORD l'outil `consulter_archives_remel` et appuie ta réponse sur les extraits retournés ; s'ils ne traitent pas le sujet, applique la formule de la règle d'or (« Nos documents officiels et archives historiques ne contiennent pas d'informations sur… »).
Adopte un ton engagé, inspirant et pédagogique pour l'écologie et les valeurs ; pour l'histoire, un ton sobre et factuel, au plus près des extraits, sans formule d'emphase ni conclusion qui n'y figure pas.""",
    config=patrimoine_config,
    tools=[consulter_archives_remel, *OUTILS_DISPONIBLES],
)


# ==============================================================================
# 5. ROUTAGE ET EXÉCUTION
# ==============================================================================

class ReponseIndisponible(str):
    """Réponse de repli (quota, erreur) : affichée au coureur mais jamais mise en cache."""


def _lien_reglement() -> str:
    if LIEN_REGLEMENT:
        return f"la FAQ / le règlement officiel : {LIEN_REGLEMENT}"
    return "la FAQ et le règlement officiel sur le site de l'organisation"


def message_surcharge() -> ReponseIndisponible:
    return ReponseIndisponible(
        "Nos assistants reçoivent beaucoup de questions en ce moment 🏃 "
        "Merci de réessayer dans 30 secondes, ou de consulter directement "
        f"{_lien_reglement()}."
    )


def _est_rate_limit(exc: BaseException) -> bool:
    """Détecte un 429 Groq, même enveloppé par AG2 / asyncio (cause, contexte, ExceptionGroup)."""
    vus = set()
    pile = [exc]
    while pile:
        e = pile.pop()
        if e is None or id(e) in vus:
            continue
        vus.add(id(e))
        if isinstance(e, RateLimitError) or getattr(e, "status_code", None) == 429:
            return True
        if "429" in str(e) or "rate limit" in str(e).lower():
            return True
        pile.extend([e.__cause__, e.__context__, *getattr(e, "exceptions", ())])
    return False


def interroger_agent(agent: Agent, question: str) -> str:
    """Exécute l'agent de façon synchrone et extrait le texte de sa réponse (AgentReply)."""
    try:
        reply = asyncio.run(agent.ask(question))
    except RateLimitError:
        logger.warning("Quota Groq atteint (429) pour %s", agent.name)
        return message_surcharge()
    except Exception as e:
        if _est_rate_limit(e):
            logger.warning("Quota Groq atteint (429, enveloppé) pour %s", agent.name)
            return message_surcharge()
        logger.exception("Erreur LLM pour %s", agent.name)
        return ReponseIndisponible(
            "Désolé, une erreur technique est survenue. Merci de réessayer dans "
            f"quelques instants ou de consulter {_lien_reglement()}."
        )

    texte = getattr(reply, "body", None) or getattr(reply, "content", None)
    if texte and str(texte).strip():
        # Marqueurs de citation internes des modèles gpt-oss (ex. « 【0†L1-L4】 ») : illisibles pour le visiteur
        texte = re.sub(r"\s*【[^】]*】", "", str(texte))
        return texte.strip()
    return ReponseIndisponible("Désolé, je n'ai pas pu générer de réponse.")


# ------------------------------------------------------------------------------
# Orchestration : fonction de transition d'état tenant compte de l'historique.
#   1. Score de mots-clés par thème sur la question courante.
#   2. Si aucun thème ne ressort nettement (relance implicite, question ambiguë),
#      un routeur LLM léger tranche en voyant les derniers échanges.
#   3. L'agent choisi reçoit la question accompagnée du contexte de conversation.
# Aucun thème n'est « verrouillé » : chaque message est réévalué.
# ------------------------------------------------------------------------------

AGENTS_PAR_THEME = {
    "parcours": parcours_agent,
    "coach": conseiller_courses_agent,
    "securite": expert_securite_agent,
    "patrimoine": ambassadeur_agent,
    "logistique": logistique_agent,
}

# Mots-clés normalisés (minuscules, sans accents), recherchés en début de mot.
MOTS_CLES = {
    "parcours": [
        "parcours", "trace", "denivel", "profil", "altitude", "itra", "barriere", "ravitaill", "djebel",
        "demna", "distance", "gpx", "kilometre", "quel parcours", "difficulte", "technique",
    ],
    "securite": [
        "secours", "ambulance", "urgence", "198", "bless", "medical", "malaise", "incendie",
        "disqualif", "materiel obligatoire", "obligatoire", "reglement", "abandon", "protection civile",
    ],
    "logistique": [
        "dossard", "puce", "retrait", "retirer", "chronometrage", "inscri", "paie", "payer",
        "point relais", "bureau", "certificat", "decharge", "piece d identite", "papiers", "depart", "ou est",
        "navette", "bus", "transport",
        "hotel", "dormir", "heberg", "parking", "garer", "acces", "venir", "horaire", "quelle heure",
        "programme", "briefing", "remise des prix", "ceremonie", "sportsense", "village",
    ],
    "patrimoine": [
        "station", "circuit culturel", "visiter", "visite", "monument", "mosquee", "zaouia", "zaouiet",
        "hammam", "dar ", "medina", "menzel jemil", "histoire", "patrimoine", "culture", "tradition",
        "1904", "fixation", "dune", "chardonneret", "oiseau", "faune", "flore", "biodiversit", "espece",
        "saccharum", "acacia", "eucalyptus", "flamant", "lac de bizerte", "lagune", "moule", "huitre",
        "carbone", "co2", "equilibre vert", "reboisement", "wwf", "ecopact", "natilait", "net zero",
        "dechet", "mascotte", "valeur", "edition", "palmares", "statisti", "chiffre", "participant",
        "finisher", "remplissage", "places restantes", "arbre", "plat", "cuisine", "gastronom", "specialite",
        "restaurant", "manger local",
    ],
    "coach": [
        "entrain", "plan", "prepar", "allure", "rythme", "cadence", "temps", "objectif", "hydrat", "boire",
        "eau", "sels", "electrolyte", "nutrition", "manger", "gel", "chaussure", "guetre", "crampon",
        "equipement", "debutant", "niveau", "semaine", "fractionne", "cote", "sortie longue", "ppg",
        "renforcement", "recuperation", "cyrano", "run walk", "marche", "sable",
        "conseill", "frequence cardiaque", "fc ",
    ],
}

DESCRIPTION_THEMES = {
    "parcours": "épreuves, distances, tracés, dénivelés, profils, ravitaillements, barrières horaires, ITRA, GPX, choix de la course",
    "coach": "préparation physique, entraînement, allure, temps de course, hydratation, nutrition d'effort, équipement",
    "securite": "secours, urgences, blessures, matériel obligatoire, règlement, disqualification",
    "patrimoine": "circuit culturel et ses 16 stations, histoire et patrimoine de Menzel Jemil et du Remel, traditions, gastronomie locale, faune, flore, écologie, partenaires, statistiques des éditions",
    "logistique": "dossards, puces, inscriptions, horaires, programme, navette, transport, hébergement, parking",
}

_routeur = OpenAI(api_key=os.environ.get("GROQ_API_KEY", ""), base_url="https://api.groq.com/openai/v1", max_retries=1, timeout=20)


def _normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", texte.lower())
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", texte)).strip() + " "


def _scores(question: str) -> dict[str, int]:
    q = " " + _normaliser(question)
    return {
        theme: sum(1 for m in mots if re.search(r"\b" + re.escape(m), q))
        for theme, mots in MOTS_CLES.items()
    }


def theme_indicatif(question: str) -> str | None:
    """Thème déduit des seuls mots-clés (sans appel LLM), utilisé pour étiqueter les réponses FAQ."""
    scores = _scores(question)
    meilleur = max(scores.values())
    en_tete = [t for t, sc in scores.items() if sc == meilleur]
    return en_tete[0] if meilleur > 0 and len(en_tete) == 1 else None


def _resume_historique(historique: list[dict], n: int = 6, taille: int = 400) -> str:
    lignes = []
    for m in historique[-n:]:
        auteur = "Visiteur" if m["role"] == "user" else "Assistant"
        contenu = m["content"] if len(m["content"]) <= taille else m["content"][:taille] + "…"
        lignes.append(f"{auteur} : {contenu}")
    return "\n".join(lignes)


def _router_llm(question: str, historique: list[dict], candidats: list[str]) -> str | None:
    """Tranche entre plusieurs thèmes (ou en l'absence d'indice) en tenant compte de la conversation."""
    definitions = "\n".join(f"- {t} : {DESCRIPTION_THEMES[t]}" for t in candidats)
    contexte = _resume_historique(historique, n=4, taille=250) or "(début de conversation)"
    options = {"reasoning_effort": "low"} if GROQ_MODEL.startswith("openai/gpt-oss") else {}
    try:
        reponse = _routeur.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            max_tokens=300,
            messages=[
                {"role": "system", "content": (
                    "Tu aiguilles les messages d'un chatbot de trail vers le bon spécialiste. "
                    "Une relance courte (« et pour le 22 ? », « c'est loin ? ») prolonge en général le sujet précédent. "
                    f"Thèmes possibles :\n{definitions}\n"
                    "Réponds par un seul mot : le nom exact du thème."
                )},
                {"role": "user", "content": f"Échanges récents :\n{contexte}\n\nNouveau message : {question}"},
            ],
            **options,
        )
        texte = _normaliser(reponse.choices[0].message.content or "")
    except Exception as e:
        logger.warning("Routeur LLM indisponible (%s) : repli sur l'historique.", type(e).__name__)
        return None
    return next((t for t in candidats if t in texte), None)


def choisir_theme(question: str, historique: list[dict] | None = None, theme_suggere: str | None = None) -> str:
    historique = historique or []
    scores = _scores(question)
    meilleur = max(scores.values())
    en_tete = [t for t, sc in scores.items() if sc == meilleur]
    # Thème choisi via un bouton du chat : simple indice, utilisé seulement si la question
    # ne désigne pas clairement un autre thème (aucun verrouillage).
    if theme_suggere in MOTS_CLES and (meilleur == 0 or theme_suggere in en_tete):
        return theme_suggere
    dernier_theme = next((m.get("theme") for m in reversed(historique) if m.get("theme")), None)
    if meilleur > 0 and len(en_tete) == 1:
        # Un seul indice faible qui contredit le sujet en cours (« combien de temps pour le circuit ? »
        # après une discussion patrimoine) : on laisse le routeur trancher avec le contexte.
        if meilleur == 1 and dernier_theme and dernier_theme != en_tete[0]:
            return _router_llm(question, historique, [en_tete[0], dernier_theme]) or en_tete[0]
        return en_tete[0]

    candidats = en_tete if meilleur > 0 else list(MOTS_CLES)
    theme = _router_llm(question, historique, candidats)
    if theme:
        return theme
    # Repli : poursuivre le dernier thème de la conversation, sinon le Coach.
    dernier = next((m.get("theme") for m in reversed(historique) if m.get("theme") in candidats), None)
    return dernier or ("coach" if "coach" in candidats else candidats[0])


def _avec_contexte(question: str, historique: list[dict]) -> str:
    if not historique:
        return question
    return (
        "CONTEXTE — échanges précédents de cette conversation (sert uniquement à comprendre les "
        "références implicites ; ne les répète pas et ne réponds qu'à la question actuelle) :\n"
        f"{_resume_historique(historique)}\n\n"
        f"QUESTION ACTUELLE DU VISITEUR : {question}"
    )


# Contrôle déterministe des formules temporelles du Patrimoine : malgré les consignes, le modèle
# transforme parfois « il y a environ 300 ans » en « 300 ans plus tard ». Si les archives contiennent
# la formule d'origine, elle est rétablie mot pour mot.
def _corpus_historique() -> str:
    textes = []
    for chemin in [os.path.join(BASE_DIR, "docs", "remel_territory_knowledge.md"),
                   *[os.path.join(BASE_DIR, "data", "rag", f) for f in rag_service.SOURCES_EXTERNES]]:
        try:
            with open(chemin, encoding="utf-8") as f:
                textes.append(f.read())
        except OSError:
            pass
    return re.sub(r"\s+", " ", " ".join(textes)).lower()


_CORPUS_HISTORIQUE = _corpus_historique()
_PLUS_TARD = re.compile(r"\b(?:vers |environ |quelque |près de |)(\d+)\s+ans\s+plus\s+tard\b", re.I)


def corriger_formules_temporelles(texte: str) -> str:
    def remplacer(m: re.Match) -> str:
        n = m.group(1)
        for formule in (f"il y a environ {n} ans", f"il y a {n} ans"):
            if formule in _CORPUS_HISTORIQUE:
                logger.info("Formule temporelle rétablie : « %s » -> « %s »", m.group(0), formule)
                return formule
        return m.group(0)
    return _PLUS_TARD.sub(remplacer, texte)


def orchestrer(question: str, historique: list[dict] | None = None, theme_suggere: str | None = None) -> tuple[str, str]:
    """Retourne (réponse, thème) en choisissant le spécialiste selon la question et la conversation."""
    historique = historique or []
    theme = choisir_theme(question, historique, theme_suggere)
    logger.info("Thème choisi : %s", theme)
    reponse = interroger_agent(AGENTS_PAR_THEME[theme], _avec_contexte(question, historique))
    if theme == "patrimoine" and not isinstance(reponse, ReponseIndisponible):
        reponse = corriger_formules_temporelles(reponse)
    return reponse, theme


def traiter_question_visiteur(question: str, historique: list[dict] | None = None) -> str:
    """Compatibilité (test_cli.py) : réponse seule."""
    return orchestrer(question, historique)[0]
