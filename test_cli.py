"""Tests de comportement du chatbot (appels réels au modèle Groq).

Usage : uv run python test_cli.py
"""
import re

from agents import orchestrer

echecs = 0


def verifier(nom: str, condition: bool, detail: str = "") -> None:
    global echecs
    echecs += 0 if condition else 1
    print(f"  {'OK ' if condition else 'ÉCHEC'} {nom}{' — ' + detail if detail and not condition else ''}")


# Test 1 : réponse depuis la documentation locale (charte zéro déchet)
print("\n=== TEST 1 : DOC LOCALE (Zéro déchet) ===")
q1 = "Est-ce qu'on a le droit aux gobelets jetables sur les ravitos ?"
r1, theme1 = orchestrer(q1)
print(f"Q: {q1}\nR [{theme1}]: {r1}")
verifier("gobelet réutilisable rappelé comme obligatoire", bool(re.search(r"obligatoire|interdit|aucun gobelet", r1, re.I)))

# Test 2 : hébergement — refus d'inventer des hôtels (recherche web désactivée)
print("\n=== TEST 2 : HÉBERGEMENT (aucun hôtel inventé) ===")
q2 = "Donne-moi 2 ou 3 noms d'hôtels connus sur la corniche de Bizerte."
r2, theme2 = orchestrer(q2)
print(f"Q: {q2}\nR [{theme2}]: {r2}")
verifier("aiguillé vers la Logistique", theme2 == "logistique", f"thème obtenu : {theme2}")
verifier("mentionne la formule d'hébergement officielle à 30 DT", bool(re.search(r"30\s*DT", r2)))
verifier("renvoie vers la corniche de Bizerte ou refuse de citer des hôtels",
         bool(re.search(r"corniche|ne (?:contiennent|citent) pas|aucun", r2, re.I)))
# Un nom propre après « Hôtel / Résidence / Resort … » trahit un établissement inventé ou non conventionné.
# (insensible à la casse pour le mot « hôtel », mais le nom doit commencer par une majuscule)
noms = re.findall(r"\b(?i:h[oô]tels?|r[ée]sidences?|resorts?|maisons? d'h[oô]tes)\s+(?:(?i:el|la|le|les)\s+)?([A-ZÀ-Ý][\w'-]+)", r2)
verifier("aucun nom d'établissement cité", not noms, f"noms détectés : {noms}")

print(f"\n{'Tous les tests passent.' if not echecs else f'{echecs} vérification(s) en échec.'}")
raise SystemExit(1 if echecs else 0)
