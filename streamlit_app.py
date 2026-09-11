"""Dashboard local : make dashboard."""

from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard import charts
from src.dashboard.data import (
    CONFIG_COLUMNS, PROJECT_ROOT, DashboardDataError, common_questions,
    database_path, export_csv, file_signature, load_results, summarize,
)


st.set_page_config(page_title="Trivia Lab · Benchmark IA", page_icon="◈", layout="wide")
st.html(f"<style>{(PROJECT_ROOT / 'src/dashboard/styles.css').read_text()}</style>")


@st.cache_data(show_spinner=False, max_entries=3)
def cached_results(path: str, signature: tuple) -> pd.DataFrame:
    # La signature est un argument de cache, sans conserver de connexion DuckDB.
    return load_results(Path(path))


def number(value: float, digits: int = 0) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}".replace(",", "\u202f").replace(".", ",")


def title(text: str, caption: str | None = None) -> None:
    st.html(f'<div class="section-title">{html.escape(text)}</div>')
    if caption:
        st.caption(caption)


def stat(label: str, value: str, note: str) -> None:
    st.html(
        f'<div class="stat"><div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{html.escape(value)}</div>'
        f'<div class="stat-note">{html.escape(note)}</div></div>'
    )


def plot(figure, key: str) -> None:
    st.plotly_chart(figure, width="stretch", theme=None, key=key, config={"displayModeBar": False})


def main() -> None:
    with st.sidebar:
        st.html('<div class="side-brand"><span class="mark">t.</span><strong>TRIVIA LAB</strong><p>Observatoire des modèles locaux</p></div>')
        if st.button("Actualiser les données", icon=":material/refresh:", width="stretch"):
            cached_results.clear()
        st.caption("PÉRIMÈTRE DE COMPARAISON")

    st.html('<div class="masthead"><span class="wordmark">TRIVIA / LAB</span><span class="edition">OPEN TRIVIA DATABASE · BENCHMARK LOCAL</span></div>')
    st.html('<div class="hero"><div class="eyebrow">MESURER · COMPARER · COMPRENDRE</div><h1>Les modèles à <em>l’épreuve.</em></h1><p>Qui répond le mieux, à quelle vitesse et sur quels sujets ? Explorez les résultats de votre benchmark.</p></div>')

    path = database_path()
    try:
        with st.spinner("Lecture des résultats du benchmark…"):
            data = cached_results(str(path), file_signature(path))
    except DashboardDataError as error:
        st.error(str(error))
        st.code("make dbt-build\nmake dashboard", language="bash")
        st.stop()
    if data.empty:
        st.info("Aucun résultat disponible. Lance un benchmark, puis `make dbt-build`.")
        st.stop()

    built_at = data["gold_built_at"].max()
    silver_root = Path(os.environ.get("BI_BENCHMARK_ROOT", PROJECT_ROOT))
    silver = silver_root / "data/silver/benchmark_results.parquet"
    if silver.exists() and silver.stat().st_mtime > built_at.timestamp():
        st.warning("De nouveaux résultats Silver sont disponibles. Lance `make dbt-build`, puis clique sur « Actualiser les données ».")

    # Un tuple réel plutôt que trois listes indépendantes : aucune combinaison fictive.
    configurations = list(data[CONFIG_COLUMNS].drop_duplicates().sort_values(CONFIG_COLUMNS).itertuples(index=False, name=None))
    with st.sidebar:
        configuration = st.selectbox(
            "Configuration", configurations, key="configuration",
            format_func=lambda value: f"{value[0]} · T={value[1]:g} · seed {value[2]}",
            help="Une seule configuration à la fois pour comparer les modèles à paramètres identiques.",
        )
        scoped = data.loc[data[CONFIG_COLUMNS].eq(configuration).all(axis=1)]
        available_models = sorted(scoped["model_name"].unique())
        models = st.multiselect("Modèles", available_models, default=available_models, key="models")
        paired = st.toggle("Questions communes uniquement", value=True, key="paired",
                           help="Conserve les questions tentées par chacun des modèles sélectionnés, erreurs incluses.")
        st.divider()
        categories = st.multiselect("Catégories", sorted(scoped["category"].unique()), key="categories",
                                    placeholder="Toutes les catégories")
        difficulties = st.multiselect("Difficultés", [x for x in ("easy", "medium", "hard") if x in set(scoped["difficulty"])],
                                      key="difficulties", placeholder="Toutes les difficultés")
        types = st.multiselect("Types de question", sorted(scoped["question_type"].unique()), key="types",
                              placeholder="Tous les types")
        st.caption("Filtres de catégorie, difficulté et type vides = tout inclure.")
        st.divider()
        st.caption(f"Gold actualisée le {built_at.strftime('%d/%m/%Y à %H:%M UTC')}\n\n{number(len(data))} résultats disponibles")

    if not models:
        st.info("Sélectionne au moins un modèle dans le panneau de gauche.")
        st.stop()
    filtered = scoped.loc[scoped["model_name"].isin(models)].copy()
    for column, selected in (("category", categories), ("difficulty", difficulties), ("question_type", types)):
        if selected:
            filtered = filtered.loc[filtered[column].isin(selected)]
    before_pairing = len(filtered)
    if paired:
        filtered = common_questions(filtered, models)
    if filtered.empty:
        st.info("Aucun résultat pour ce périmètre. Élargis les filtres ou désactive « Questions communes uniquement ».")
        st.stop()

    labels = {model: model.split("/")[-1] for model in available_models}
    if len(set(labels.values())) != len(labels):
        labels = {model: model for model in available_models}
    colors = {labels[model]: charts.PALETTE[i % len(charts.PALETTE)] for i, model in enumerate(available_models)}
    filtered["model_label"] = filtered["model_name"].map(labels)
    summary = summarize(filtered)
    summary["model_label"] = summary["model_name"].map(labels)

    mode = st.radio("Indicateur de précision", ["Accuracy globale", "Sur réponses valides"], horizontal=True, key="metric",
                    help="Globale = correctes / tentatives. Sur réponses valides = correctes / réponses interprétables.")
    metric = "overall_accuracy" if mode == "Accuracy globale" else "valid_answer_accuracy"
    ranked = summary.sort_values([metric, "model_name"], ascending=[False, True], na_position="last")
    best = ranked.iloc[0]
    score = f"{number(best[metric], 2)} %" if pd.notna(best[metric]) else "—"
    best_models = summary.loc[summary[metric].eq(best[metric]), "model_label"].tolist()
    best_note = " · ".join(best_models) if len(best_models) == 1 else f"{len(best_models)} modèles ex æquo"
    if not best_models:
        best_note = "Aucune réponse valide"
    for column, args in zip(st.columns(4), (
        ("Modèles comparés", number(summary["model_name"].nunique()), "Même prompt, température et seed"),
        ("Questions du périmètre", number(filtered["question_id"].nunique()), f"{number(len(filtered))} tentatives analysées"),
        ("Meilleure accuracy", score, best_note),
        ("Réponses valides", f"{number(100 * filtered['status'].eq('success').mean(), 2)} %", "Réponse interprétable, correcte ou non"),
    )):
        with column:
            stat(*args)

    if paired:
        st.caption(f"Comparaison sur les mêmes questions · {number(before_pairing - len(filtered))} tentatives hors intersection exclues.")
    else:
        st.warning("Périmètre libre : les modèles peuvent avoir répondu à des questions différentes. Le classement est descriptif.")
    if metric == "valid_answer_accuracy" and summary["valid_answers"].eq(0).any():
        st.info("Un modèle sans réponse valide a une accuracy indéfinie (—) et n'apparaît pas dans les graphiques de précision.")

    overview, dimensions, execution, answers = st.tabs(["Comparatif", "Par thème", "Exécution", "Explorer les réponses"])
    with overview:
        left, right = st.columns([1.15, 1])
        with left:
            with st.container(border=True):
                title("Précision des modèles", "Part de bonnes réponses · survolez les barres pour voir les effectifs.")
                plot(charts.ranking(summary, metric, colors), "ranking")
        with right:
            with st.container(border=True):
                title("Précision × temps de réponse", "En haut à gauche : une précision élevée avec un temps de réponse court.")
                plot(charts.tradeoff(summary, metric, colors), "tradeoff")
        if len(summary) == 2 and paired:
            paired_scores = filtered.pivot(index="question_id", columns="model_name", values="ai_correct")
            a, b = summary["model_name"].tolist()
            only_a = int((paired_scores[a] & ~paired_scores[b]).sum())
            only_b = int((~paired_scores[a] & paired_scores[b]).sum())
            both = int((paired_scores[a] & paired_scores[b]).sum())
            st.html(f'<div class="insight"><strong>Face à face.</strong> {number(both)} questions réussies par les deux modèles ; '
                    f'{number(only_a)} réussies uniquement par {html.escape(labels[a])}, '
                    f'{number(only_b)} uniquement par {html.escape(labels[b])}.</div>')
        title("Les chiffres, côte à côte")
        visible = ["model_name", "total_questions", "correct_answers", "overall_accuracy", "valid_answer_accuracy", "avg_response_time", "valid_answer_rate"]
        st.dataframe(ranked[visible], hide_index=True, width="stretch", column_config={
            "model_name": "Modèle", "total_questions": "Tentatives", "correct_answers": "Correctes",
            "overall_accuracy": st.column_config.ProgressColumn("Accuracy globale", min_value=0, max_value=100, format="%.2f%%"),
            "valid_answer_accuracy": st.column_config.NumberColumn("Accuracy / valides", format="%.2f%%"),
            "avg_response_time": st.column_config.NumberColumn("Temps moyen", format="%.3f s"),
            "valid_answer_rate": st.column_config.NumberColumn("Réponses valides", format="%.2f%%"),
        })
        st.download_button("Exporter la comparaison · CSV", export_csv(ranked.drop(columns="model_label")),
                           "comparaison_benchmark.csv", "text/csv", key="export_summary")

    with dimensions:
        dimension_label = st.radio("Analyser par", ["Catégorie", "Difficulté", "Type de question"], horizontal=True, key="dimension")
        dimension = {"Catégorie": "category", "Difficulté": "difficulty", "Type de question": "question_type"}[dimension_label]
        by_dimension = summarize(filtered, dimension)
        by_dimension["model_label"] = by_dimension["model_name"].map(labels)
        title(f"Précision par {dimension_label.lower()}", "Même échelle de 0 à 100 % pour tous les modèles. Les effectifs sont affichés au survol.")
        plot(charts.breakdown(by_dimension, dimension, metric, colors), "breakdown")
        st.download_button("Exporter cette analyse · CSV", export_csv(by_dimension.drop(columns="model_label")),
                           f"benchmark_par_{dimension}.csv", "text/csv", key="export_dimension")

    with execution:
        title("Comment se terminent les tentatives ?", "Une réponse incorrecte reste valide si le choix du modèle a pu être interprété.")
        plot(charts.quality(summary), "quality")
        title("Latence et consommation")
        st.dataframe(summary[["model_name", "invalid_answers", "technical_errors", "avg_response_time", "median_response_time", "p95_response_time", "avg_prompt_tokens", "avg_completion_tokens"]],
                     hide_index=True, width="stretch", column_config={
            "model_name": "Modèle", "invalid_answers": "Invalides", "technical_errors": "Erreurs techniques",
            **{column: st.column_config.NumberColumn(label, format="%.3f s") for column, label in (
                ("avg_response_time", "Temps moyen"), ("median_response_time", "Médiane"), ("p95_response_time", "P95"))},
            "avg_prompt_tokens": st.column_config.NumberColumn("Tokens entrée / réponse", format="%.2f"),
            "avg_completion_tokens": st.column_config.NumberColumn("Tokens sortie / réponse", format="%.2f"),
        })
        st.caption("Les temps incluent les tentatives échouées, les relances automatiques et leur attente éventuelle. "
                   "Le matériel, le chargement du modèle et son état de chauffe peuvent influencer la latence. "
                   "Les moyennes de tokens ignorent les valeurs non fournies par le serveur.")

    with answers:
        title("Revenir à la question", "Ces filtres s'appliquent uniquement à l'explorateur et à son export.")
        search_column, outcome_column = st.columns([2, 1])
        search = search_column.text_input("Rechercher dans les questions", placeholder="Un mot, un sujet…", key="search")
        outcome = outcome_column.selectbox("Résultat", ["Tous", "Correcte", "Incorrecte", "Invalide", "Erreur technique"], key="outcome")
        rows = filtered.copy()
        rows["result_label"] = rows["status"].map({"invalid_answer": "Invalide", "error": "Erreur technique", "success": "Incorrecte"})
        rows.loc[rows["ai_correct"], "result_label"] = "Correcte"
        if search:
            rows = rows.loc[rows["question"].str.contains(search, case=False, regex=False, na=False)]
        if outcome != "Tous":
            rows = rows.loc[rows["result_label"].eq(outcome)]
        st.caption(f"{number(len(rows))} réponses correspondent à la recherche.")
        if rows.empty:
            st.info("Aucune réponse ne correspond à cette recherche.")
        else:
            display_columns = ["model_name", "category", "difficulty", "question", "ai_answer", "correct_choice", "result_label", "response_time"]
            selection = st.dataframe(rows[display_columns].reset_index(drop=True), hide_index=True, width="stretch", height=350,
                                    on_select="rerun", selection_mode="single-row", key="response_table", column_config={
                "model_name": "Modèle", "category": "Catégorie", "difficulty": "Difficulté",
                "question": st.column_config.TextColumn("Question", width="large"),
                "ai_answer": "Choix IA", "correct_choice": "Attendu", "result_label": "Résultat",
                "response_time": st.column_config.NumberColumn("Temps", format="%.3f s"),
            })
            indices = selection.selection.rows
            index = indices[0] if indices and indices[0] < len(rows) else 0
            record = rows.iloc[index]
            with st.expander("Détail de la réponse sélectionnée (première ligne par défaut)", expanded=True):
                st.text(record["question"])
                choices, response = st.columns(2)
                with choices:
                    st.caption("CHOIX PROPOSÉS")
                    for i, choice in enumerate(record["answer_choices"]):
                        label = chr(ord("A") + i)
                        st.text(f"{label}. {choice}" + ("  ✓" if label == record["correct_choice"] else ""))
                with response:
                    st.caption(f"RÉPONSE BRUTE · {record['model_name']}")
                    raw_answer = record["ai_answer_raw"]
                    st.code(raw_answer if pd.notna(raw_answer) else "Aucun contenu reçu", language=None)
                    if pd.notna(record["error_message"]):
                        st.text(record["error_message"])
                    st.caption(f"{record['result_label']} · {number(record['response_time'], 3)} s · {record['created_at'].strftime('%d/%m/%Y %H:%M UTC')}")
            st.download_button("Exporter les réponses filtrées · CSV", export_csv(rows.drop(columns="model_label")),
                               "reponses_benchmark.csv", "text/csv", key="export_answers")

    with st.expander("Comment lire ces résultats ?"):
        st.markdown(
            "**Accuracy globale** = réponses correctes / toutes les tentatives. Les réponses invalides et les erreurs techniques comptent comme des échecs.\n\n"
            "**Accuracy sur réponses valides** = réponses correctes / réponses avec `status = success`. "
            "Sans réponse valide, la valeur est indéfinie et affichée « — ».\n\n"
            "Les taux sont recalculés depuis les résultats individuels après filtrage, sans moyenne de pourcentages. "
            "Le périmètre commun retient les questions tentées par tous les modèles sélectionnés. "
            "Les données représentent le dernier résultat conservé pour chaque question et configuration, pas un historique de toutes les relances.\n\n"
            "Un petit écart de score ne démontre pas à lui seul qu'un modèle est meilleur en général. "
            "Ces résultats portent sur ce dataset et cette configuration."
        )
    st.html('<div class="footnote">TRIVIA LAB · OpenTDB → Silver Parquet → dbt / DuckDB → Streamlit · Consultation en lecture seule</div>')


if __name__ == "__main__":
    main()
