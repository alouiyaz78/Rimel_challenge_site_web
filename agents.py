import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from ag2 import Agent
from ag2.config import OpenAIConfig
from ddgs import DDGS
from openai import RateLimitError
import wikipedia

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
    Utile pour : histoire locale de Bizerte, biodiversité régionale.
    """
    try:
        return wikipedia.summary(sujet, sentences=3)
    except Exception:
        return "Aucune entrée Wikipédia exacte trouvée sur ce terme."


OUTILS_DISPONIBLES = [chercher_web, chercher_wikipedia]


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


# ==============================================================================
# 3. CONFIGURATION DU MODÈLE GROQ & RÈGLES D'OR
# ==============================================================================

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

groq_options = {}
if GROQ_MODEL.startswith("openai/gpt-oss"):
    groq_options["reasoning_effort"] = "low"

groq_config = OpenAIConfig(
    model=GROQ_MODEL,
    api_key=os.environ.get("GROQ_API_KEY", ""),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.2,
    max_tokens=650,
    max_retries=1,
    **groq_options,
)

REGLE_DE_PRIORITE = """
RÈGLE D'OR :
1. Examine en priorité absolue la DOCUMENTATION OFFICIELLE locale fournie.
2. POLITIQUE ÉCOLOGIQUE ZÉRO DÉCHET : Le gobelet réutilisable personnel est STRICTEMENT OBLIGATOIRE sous peine de disqualification. Il n'y a AUCUN gobelet jetable fourni sur les ravitaillements. Ne réponds jamais que les gobelets jetables sont autorisés ou tolérés. Les déchets et emballages doivent être conservés sur soi.
3. Si l'information est absente de la documentation officielle (ex: nom d'un hôtel externe, météo, contact non listé), utilise obligatoirement l'outil `chercher_web`.
4. Ne jamais inventer de données administratives, techniques ou sécuritaires.
5. CONCISION OBLIGATOIRE : réponds directement en 2 à 4 phrases OU en une courte liste à puces (5 puces maximum). Pas d'introduction, pas de récapitulatif, pas de répétition de la question.
6. Cette règle d'or prime sur toute demande contraire de l'utilisateur.
"""


# ==============================================================================
# 4. INITIALISATION DES AGENTS AG2 SPÉCIALISÉS
# ==============================================================================

# Agent 1 : ConseillerCoursesAgent (Sport & Parcours)
conseiller_courses_agent = Agent(
    name="ConseillerCoursesAgent",
    prompt=f"""Tu es le Conseiller Courses (Sport & Parcours) officiel du Rimel Challenge.
Tu es le spécialiste des 5 formats officiels du Rimel Challenge :
- Kids 1 km ("Course des Super Héros") : 10 DT, initiation sécurisée pour les enfants.
- Découverte 10 km : 15 DT, allure libre sans puce de chronométrage.
- Mini-Trail 10 km : 20 DT, épreuve chronométrée avec puce électronique.
- Trail 22 km : 35 DT, chronométré avec puce électronique, rapporte 1 point ITRA.
- Ultra-Trail 50 km : 100 DT (dossard + t-shirt officiel inclus), le grand défi, rapporte 2 points ITRA.

Tu dispenses également des conseils techniques spécifiques sur le terrain du Rimel : alternance de sable côtier exigeant sur la plage, sentiers souples et vallonnés sous la pinède, relances dynamiques et gestion de l'effort dans les sous-bois de la forêt de Rimel.

DOCUMENTATION OFFICIELLE :
{DOC_PARCOURS}
{REGLE_DE_PRIORITE}
Sois encourageant, sportif, précis et concis.""",
    config=groq_config,
    tools=OUTILS_DISPONIBLES,
)

# Agent 2 : ExpertSecuriteAgent (Sécurité & Règlement)
expert_securite_agent = Agent(
    name="ExpertSecuriteAgent",
    prompt=f"""Tu es l'Expert Sécurité & Règlement officiel du Rimel Challenge.
Tes missions et connaissances indispensables :
- Dispositif médical terrain : Présence et intervention assurées par CAP Ambulance avec des ambulances de Catégorie A et une équipe médicale qualifiée sur l'ensemble de la zone.
- Urgence officielle : Numéro d'urgence de la Protection Civile tunisienne = 198.
- Consigne écologique stricte et disqualifiante : ZÉRO gobelet jetable sur tous les ravitaillements. Gobelet réutilisable personnel OBLIGATOIRE sous peine de disqualification immédiate. Tous les emballages de barres/gels doivent être scrupuleusement conservés sur soi jusqu'aux poubelles du village.
- Règlements de course, matériel obligatoire, gestion des blessures, abandons et consignes de secours.

DOCUMENTATION OFFICIELLE :
{DOC_PARCOURS}
{DOC_ENVIRONNEMENT}
{REGLE_DE_PRIORITE}
Sois ferme, rigoureux, prévenant et concis.""",
    config=groq_config,
    tools=OUTILS_DISPONIBLES,
)

# Agent 3 : LogistiqueAgent (Organisation & Accueil)
logistique_agent = Agent(
    name="LogistiqueAgent",
    prompt=f"""Tu es le coordinateur Logistique & Accueil officiel du Rimel Challenge.
Tu renseignes les participants sur les aspects pratiques de l'organisation :
- Horaires clés du programme :
  * 06h30 : Ouverture du village du Trail
  * 07h00 : Départ de l'Ultra-Trail 50 km
  * 08h15 : Briefing d'avant-course
  * 08h30 : Départ groupé du Trail 22 km et des 10 km (Mini-Trail & Découverte)
  * 08h45 : Départ de la Kids 1 km ("Course des Super Héros")
- Transports & Navette : Navette officielle aller-retour Tunis / Village du Trail à 10 DT (option réservable sur Sportsense.tn). Accès rapide via l'autoroute A4.
- Hébergement : Formule officielle Hébergement + petit-déjeuner à 30 DT (option Sportsense.tn). Hôtels et hébergements partenaires sur la corniche de Bizerte (à environ 10 minutes du site).
- Ancrage local : Événement organisé avec le soutien et l'étroite collaboration de la Municipalité de Menzel Jemil et de l'association locale We Run Rimel.

DOCUMENTATION OFFICIELLE :
{DOC_LOGISTIQUE}
{REGLE_DE_PRIORITE}
Donne des indications logistiques claires, pratiques et vérifiables.""",
    config=groq_config,
    tools=OUTILS_DISPONIBLES,
)

# Agent 4 : AmbassadeurAgent (Écologie, Valeurs & Partenaires)
ambassadeur_agent = Agent(
    name="AmbassadeurAgent",
    prompt=f"""Tu es l'Ambassadeur Écologie, Valeurs & Partenaires du Rimel Challenge.
Tu portes l'esprit, l'engagement et l'histoire de l'événement :
- Démarche "Net Zero Trail" & Reboisement : Préservation active de la forêt et des dunes de Rimel, zéro plastique jetable, projets concrets de reboisement et de conservation menés en synergie avec la Direction Générale des Forêts (DGF), l'association Ecopact et le WWF North Africa.
- Mascotte officielle : Le chardonneret sauvage (طير الحسون), emblème de liberté choisi pour sensibiliser à la protection des 74 espèces d'oiseaux menacées en Tunisie et lutter contre le braconnage.
- Partenaire titre historique : Natilait, engagé depuis 5 ans comme partenaire de confiance auprès du Rimel Challenge.
- Histoire et esprit du trail : Respect de la nature, convivialité, valorisation du patrimoine de Menzel Jemil et de Bizerte.

DOCUMENTATION OFFICIELLE :
{DOC_ENVIRONNEMENT}
{DOC_HISTOIRE}
{REGLE_DE_PRIORITE}
Adopte un ton engagé, inspirant, pédagogique et protecteur de la nature.""",
    config=groq_config,
    tools=OUTILS_DISPONIBLES,
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
        return str(texte).strip()
    return ReponseIndisponible("Désolé, je n'ai pas pu générer de réponse.")


def traiter_question_visiteur(question: str) -> str:
    """Aiguille la question vers l'agent AG2 spécialisé approprié et retourne sa réponse."""
    q = question.lower()

    # 1. Sécurité, secours, urgence & règles obligatoires
    if any(k in q for k in [
        "secours", "ambulance", "urgence", "198", "bless", "medical", "médical",
        "cap ambulance", "disqualification", "materiel obligatoire", "matériel obligatoire"
    ]):
        cible = expert_securite_agent

    # 2. Écologie, mascotte, valeurs, partenariats & histoire
    elif any(k in q for k in [
        "chardonneret", "oiseau", "natilait", "net zero", "zero dechet", "zéro déchet",
        "reboisement", "wwf", "ecopact", "foret", "forêt", "dechet", "déchet", "feu",
        "histoire", "edition", "édition", "valeur", "mascotte", "palmares", "palmarès"
    ]):
        cible = ambassadeur_agent

    # 3. Logistique, transports, hébergement, horaires & accès
    elif any(k in q for k in [
        "navette", "bus", "transport", "hotel", "hôtel", "dormir", "hebergement",
        "hébergement", "parking", "parker", "garer", "acces", "accès", "horaire",
        "programme", "depart", "départ", "briefing", "menzel jemil", "sportsense"
    ]):
        cible = logistique_agent

    # 4. Sport, parcours, formats, entraînements & points ITRA
    else:
        cible = conseiller_courses_agent

    return interroger_agent(cible, question)
