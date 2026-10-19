"""Tests du mode dégradé des statistiques (sans réseau).

Usage : uv run python -m unittest test_stats_inscriptions
"""
import io
import json
import os
import tempfile
import unittest
import urllib.parse
from unittest import mock

import stats_inscriptions as si

LIGNES = [
    {"distance": "Rimel Challenge By Natilait - 22km", "paiement": "Carte bancaire", "nat": "TN", "sexe": "Homme", "dossard": "101"},
    {"distance": "Rimel Challenge By Natilait - 50km", "paiement": "Espèces", "nat": "FR", "sexe": "Femme", "dossard": ""},
    {"distance": "Rimel Challenge By Natilait - Trail Kids", "paiement": "Espèces", "nat": "TN", "sexe": "Homme", "dossard": None},
]
ENV = {"SUPABASE_URL": "https://client.supabase.co", "SUPABASE_KEY": "cle-client",
       "OPS_SUPABASE_URL": "https://ops.supabase.co", "OPS_SUPABASE_KEY": "cle-ops"}


class Reponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def faux_urlopen(client, ops, appels):
    """client / ops : liste de lignes renvoyée, ou exception à lever."""
    def urlopen(requete, timeout=None):
        appels.append(requete.full_url)
        reponse = ops if "ops.supabase.co" in requete.full_url else client
        if isinstance(reponse, Exception):
            raise reponse
        return Reponse(json.dumps(reponse).encode())
    return urlopen


class TestModeDegrade(unittest.TestCase):
    def setUp(self):
        si._cache.update(horodatage=0.0, donnees=None)
        dossier = tempfile.mkdtemp()
        self.instantane = os.path.join(dossier, "stats_instantane.json")
        mock.patch.object(si, "FICHIER_INSTANTANE", self.instantane).start()
        mock.patch.dict(os.environ, ENV).start()
        self.addCleanup(mock.patch.stopall)
        self.appels = []

    def stats(self, client, ops):
        with mock.patch("urllib.request.urlopen", faux_urlopen(client, ops, self.appels)):
            return si.statistiques()

    def test_base_client_utilisee_si_elle_repond(self):
        d = self.stats(LIGNES, [])
        self.assertEqual((d["source"], d["total"]), ("supabase", 3))
        self.assertFalse(any("ops.supabase.co" in u for u in self.appels))

    def test_base_client_vide_bascule_sur_copie_ops(self):
        d = self.stats([], LIGNES)
        self.assertEqual((d["source"], d["total"], d["valides"]), ("secours", 3, 1))
        self.assertEqual({c["code"]: c["total"] for c in d["courses"]}["KIDS"], 1)
        self.assertEqual(d["demographie"]["enfants"], 1)

    def test_base_client_en_erreur_bascule_sur_copie_ops(self):
        d = self.stats(OSError("401"), LIGNES)
        self.assertEqual((d["source"], d["total"]), ("secours", 3))

    def test_requete_ops_extrait_colonnes_neutres_du_jsonb(self):
        self.stats([], LIGNES)
        url = urllib.parse.unquote(next(u for u in self.appels if "ops.supabase.co" in u))
        self.assertIn("/rest/v1/registration_snapshot?select=distance:data->>distance,", url)
        self.assertIn("&data->>distance=ilike.*Rimel Challenge By Natilait*", url)
        self.assertNotIn("cin", url.lower())

    def test_tout_vide_ne_remplace_pas_l_instantane_par_des_zeros(self):
        with open(self.instantane, "w", encoding="utf-8") as f:
            json.dump({"disponible": True, "total": 1971}, f)
        d = self.stats([], [])
        self.assertEqual((d["source"], d["total"]), ("instantane", 1971))
        with open(self.instantane, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["total"], 1971)

    def test_pagination(self):
        page = [LIGNES[0]] * si.TAILLE_PAGE
        reponses = iter([page, LIGNES[:1]])
        with mock.patch("urllib.request.urlopen", lambda r, timeout=None: Reponse(json.dumps(next(reponses)).encode())):
            self.assertEqual(len(si._lire_registrations()), si.TAILLE_PAGE + 1)


if __name__ == "__main__":
    unittest.main()
