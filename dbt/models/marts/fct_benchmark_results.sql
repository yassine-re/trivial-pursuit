-- Snapshot détaillé pour les filtres croisés et l'exploration du dashboard.
-- Grain : une ligne par benchmark_id (question, modèle, prompt, configuration).
-- Matérialisé en table : tous les graphiques lisent la même version des données.
select
    *,
    current_timestamp as gold_built_at
from {{ ref('stg_benchmark_results') }}
