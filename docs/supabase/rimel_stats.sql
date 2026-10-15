-- =============================================================================
-- Rimel Challenge — sécurisation de la table registrations + statistiques agrégées
-- À exécuter dans Supabase : SQL Editor (rôle postgres).
-- =============================================================================

-- 0) DIAGNOSTIC : politiques existantes sur la table.
--    Une politique SELECT ouverte à anon/public (ex. "using (true)") explique la fuite :
--    elle doit être supprimée (drop policy "<nom>" on public.registrations;).
select policyname, roles, cmd, qual from pg_policies
where schemaname = 'public' and tablename = 'registrations';


-- 1) FERMER LA LECTURE DIRECTE AUX CLÉS PUBLIQUES
--    Avec la RLS active et sans politique SELECT pour anon, plus aucune ligne
--    (nom, prénom, email, tel, CIN…) n'est lisible via l'API avec la clé anon.
alter table public.registrations enable row level security;

--    ⚠️ Si votre formulaire d'inscription ÉCRIT dans cette table avec la clé anon,
--    autorisez uniquement l'insertion (pas la lecture) :
-- create policy "inscription_publique_insert" on public.registrations
--   for insert to anon with check (true);


-- 2) STATISTIQUES AGRÉGÉES DU RIMEL (aucune donnée personnelle, aucun montant)
--    Comptages par épreuve, moyen de paiement, dossard attribué, nationalité et sexe.
--    SECURITY DEFINER : la fonction lit la table malgré la RLS, mais ne renvoie que des totaux.
create or replace function public.rimel_stats()
returns table (distance text, paiement text, dossard_attribue boolean, nat text, sexe text, n bigint)
language sql
stable
security definer
set search_path = public
as $$
  select
    r.distance::text,
    r.paiement::text,
    nullif(trim(r.dossard::text), '') is not null,
    r.nat::text,
    r.sexe::text,
    count(*)
  from public.registrations r
  where r.distance::text ilike 'Rimel Challenge%'
  group by 1, 2, 3, 4, 5;
$$;

revoke all on function public.rimel_stats() from public;
grant execute on function public.rimel_stats() to anon, authenticated;


-- 3) VÉRIFICATIONS (avec la clé anon, depuis l'extérieur) :
--    GET  {SUPABASE_URL}/registrations?select=nom&limit=1   -> doit renvoyer []  (ou une erreur)
--    GET  {SUPABASE_URL}/rpc/rimel_stats                    -> doit renvoyer les comptages
--    Pensez à contrôler de la même façon les AUTRES tables du projet.
