from agents import traiter_question_visiteur

# Test 1 : Réponse depuis la documentation locale (Environnement)
print("\n=== TEST 1 : DOC LOCALE (Zéro déchet) ===")
q1 = "Est-ce qu'on a le droit aux gobelets jetables sur les ravitos ?"
print(f"Q: {q1}")
print("R:", traiter_question_visiteur(q1))

# Test 2 : Réponse nécessitant le Tool Web (Hôtels à Bizerte)
print("\n=== TEST 2 : OUTIL RECHERCHE WEB (Hôtels Corniche) ===")
q2 = "Donne-moi 2 ou 3 noms d'hôtels connus sur la corniche de Bizerte."
print(f"Q: {q2}")
print("R:", traiter_question_visiteur(q2))
