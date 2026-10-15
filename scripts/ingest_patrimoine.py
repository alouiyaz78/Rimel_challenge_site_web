import os
import re
import glob
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# Dossiers cibles
DOSSIER_DOCS = "docs/knowledge"
DOSSIER_RAG = "data/rag"
os.makedirs(DOSSIER_RAG, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. EXTRACTION DES ARCHIVES DE M. MOHAMED HEDI KAABORI (leplacartuel.com)
# ---------------------------------------------------------------------------
PAGES_KAABORI = [
    {
        "url": "https://leplacartuel.com/menzelini/histoire/origine.htm",
        "titre": "Origines de Menzel Jemil et archives de 1884"
    },
    {
        "url": "https://leplacartuel.com/menzelini/histoire/HistoireOrale.htm",
        "titre": "Histoire orale et mémoires locales (Cheikh الذكواني / المعزي)"
    },
    {
        "url": "https://leplacartuel.com/menzelini/histoire/HistoireMaazi.htm",
        "titre": "Périodes romaine, espagnole, ottomane et traditions locales"
    },
    {
        "url": "https://leplacartuel.com/menzelini/geo/Satfoura.htm",
        "titre": "Région historique de Satfoura (صطفورة)"
    }
]

def extraire_archives_kaabori():
    fichier_sortie = os.path.join(DOSSIER_RAG, "archives_kaabori_menzel_jemil.md")
    print("\n" + "="*50)
    print(" [1/2] Extraction des archives de M. Mohamed Hedi Kaabori...")
    print("="*50)

    with open(fichier_sortie, "w", encoding="utf-8") as f:
        f.write("# Archives Historiques de Menzel Jemil\n\n")
        f.write("**Source documentaire & Propriété intellectuelle :**\n")
        f.write("- **Auteur :** M. Mohamed Hedi Kaabori (Professeur d'histoire à Menzel Jemil)\n")
        f.write("- **Plateforme :** *leplacartuel.com*\n")
        f.write("- **Consigne de citation :** Toute réponse utilisant ces données doit mentionner : "
                "« D'après les recherches de M. Mohamed Hedi Kaabori (leplacartuel.com) ».\n\n")
        f.write("---\n\n")

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        for page in PAGES_KAABORI:
            url = page["url"]
            titre = page["titre"]
            print(f" -> Téléchargement : {titre} ({url})")
            try:
                res = requests.get(url, headers=headers, timeout=12)
                res.encoding = res.apparent_encoding or "utf-8"
                soup = BeautifulSoup(res.text, "html.parser")

                for element in soup(["script", "style", "nav", "footer", "iframe"]):
                    element.decompose()

                texte = soup.get_text(separator="\n", strip=True)
                texte_propre = re.sub(r'\n{3,}', '\n\n', texte)

                f.write(f"## {titre}\n")
                f.write(f"**URL Source :** {url}\n\n")
                f.write(texte_propre + "\n\n---\n\n")
                print(f"    Succès ({len(texte_propre)} caractères)")

            except Exception as e:
                print(f"    Erreur lors de l'extraction de {url} : {e}")

    print(f" Fichier Kaabori généré : {fichier_sortie}")


# ---------------------------------------------------------------------------
# 2. EXTRACTION CIBLÉE DU PDF DE LA THÈSE (Mohamed Ali Hbaieb)
# ---------------------------------------------------------------------------
TERMES_CIBLES = [
    "menzel djemil", "menzel jemil", "manzil jamil", "منزل جميل",
    "rimel", "r'mel", "رمال",
    "satfoura", "setfoura", "صطفورة",
    "tinja", "bizerte"
]

def chercher_fichier_pdf_these():
    """Cherche automatiquement un fichier PDF dans le dossier docs/."""
    candidats = glob.glob(os.path.join(DOSSIER_DOCS, "*.pdf"))
    for c in candidats:
        if "hbaieb" in c.lower() or "these" in c.lower():
            return c
    return candidats[0] if candidats else None

def extraire_these_hbaieb():
    print("\n" + "="*50)
    print(" [2/2] Extraction de la thèse de Mohamed Ali Hbaieb...")
    print("="*50)

    pdf_path = chercher_fichier_pdf_these()
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"⚠️ Aucun fichier PDF trouvé dans `{DOSSIER_DOCS}/`.")
        print(f"   Veuillez y copier votre thèse (ex. `{DOSSIER_DOCS}/these_hbaieb.pdf`) et relancer.")
        return

    print(f" -> PDF détecté : {pdf_path}")
    fichier_sortie = os.path.join(DOSSIER_RAG, "these_hbaieb_bizerte.md")

    reader = PdfReader(pdf_path)
    nb_pages = len(reader.pages)
    print(f" -> Lecture de {nb_pages} pages en cours...")

    extraits = []

    for i, page in enumerate(reader.pages):
        texte = page.extract_text() or ""
        texte_lower = texte.lower()

        # On cible prioritairement Menzel Jemil, Rimel, Satfoura
        if any(mot in texte_lower for mot in ["menzel", "manzil", "rimel", "satfoura", "منزل"]):
            texte_nettoye = re.sub(r'\n+', '\n', texte.strip())
            extraits.append({
                "page": i + 1,
                "texte": texte_nettoye
            })

    print(f" -> {len(extraits)} page(s) pertinente(s) trouvée(s) pour Menzel Jemil / Rimel.")

    with open(fichier_sortie, "w", encoding="utf-8") as f:
        f.write("# Thèse Historique Régionale - Extraits Menzel Jemil & Rimel\n\n")
        f.write("**Référence académique & Attribution obligatoire :**\n")
        f.write("- **Auteur :** Mohamed Ali HBAIEB\n")
        f.write("- **Type :** Thèse de doctorat en histoire (soutenue le 13/02/2009)\n")
        f.write("- **Direction de thèse :** Pierre GUICHARD (Université Lumière-Lyon 2 - UMR 5648) "
                "et Mounira CHAPOUTOT-REMADI (Université de Tunis)\n")
        f.write("- **Consigne de citation :** Toute réponse utilisant ces données doit mentionner : "
                "« D'après la thèse de doctorat de Mohamed Ali Hbaieb (Université de Tunis / Lyon 2 UMR 5648) ».\n\n")
        f.write("---\n\n")

        for item in extraits:
            f.write(f"### Extrait de la Page {item['page']}\n\n")
            f.write(item["texte"] + "\n\n---\n\n")

    print(f" Fichier Thèse généré : {fichier_sortie}")


# ---------------------------------------------------------------------------
# EXÉCUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    extraire_archives_kaabori()
    extraire_these_hbaieb()
    print("\n Tout est prêt dans `data/rag/` pour réindexer le RAG !")