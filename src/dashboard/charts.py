"""Graphiques Plotly avec une échelle et une palette cohérentes."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#176459", "#cf7646", "#507b9c", "#92753a", "#8c6085", "#738743"]
LABELS = {
    "model_label": "Modèle", "overall_accuracy": "Accuracy globale (%)",
    "valid_answer_accuracy": "Accuracy sur réponses valides (%)",
    "avg_response_time": "Temps moyen (s)", "total_questions": "Tentatives",
    "category": "Catégorie", "difficulty": "Difficulté", "question_type": "Type",
    "avg_prompt_tokens": "Tokens d'entrée / réponse",
    "avg_completion_tokens": "Tokens de sortie / réponse",
}


def finish(figure: go.Figure, height: int = 320) -> go.Figure:
    figure.update_layout(
        template="plotly_white", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Avenir Next, Trebuchet MS, sans-serif", color="#243d35", size=12),
        margin=dict(l=10, r=25, t=25, b=15),
        legend=dict(title=None, orientation="h", y=-0.2, x=0),
        hoverlabel=dict(bgcolor="#fffefa", font_color="#243d35"),
    )
    figure.update_xaxes(gridcolor="#e1e5dc", zeroline=False, title_font_size=11)
    figure.update_yaxes(gridcolor="#e1e5dc", zeroline=False, title_font_size=11)
    return figure


def ranking(summary: pd.DataFrame, metric: str, colors: dict) -> go.Figure:
    frame = summary.sort_values(metric, ascending=True)
    figure = px.bar(
        frame, x=metric, y="model_label", orientation="h", color="model_label",
        color_discrete_map=colors, text=metric, labels=LABELS,
        hover_data={"total_questions": True, "correct_answers": True, metric: ":.2f"},
    )
    figure.update_traces(texttemplate="%{x:.2f}%", textposition="outside", cliponaxis=False, width=0.48)
    finish(figure, max(260, len(frame) * 65 + 110))
    figure.update_layout(showlegend=False)
    figure.update_xaxes(range=[0, 110], tickvals=[0, 25, 50, 75, 100], ticksuffix=" %", title=None)
    figure.update_yaxes(title=None, showgrid=False)
    return figure


def tradeoff(summary: pd.DataFrame, metric: str, colors: dict) -> go.Figure:
    figure = px.scatter(
        summary, x="avg_response_time", y=metric, color="model_label",
        color_discrete_map=colors, labels=LABELS, hover_name="model_name",
        hover_data={"avg_response_time": ":.3f", metric: ":.2f", "total_questions": True},
    )
    figure.update_traces(marker=dict(size=18, line=dict(width=2, color="#fffefa")))
    finish(figure, max(260, len(summary) * 65 + 110))
    figure.update_xaxes(range=[0, max(float(summary["avg_response_time"].max()) * 1.35, 0.1)])
    figure.update_yaxes(range=[0, 100], ticksuffix=" %")
    return figure


def breakdown(summary: pd.DataFrame, dimension: str, metric: str, colors: dict) -> go.Figure:
    if dimension == "category":
        scores = summary.pivot(index=dimension, columns="model_label", values=metric).sort_index()
        counts = summary.pivot(index=dimension, columns="model_label", values="total_questions").reindex_like(scores)
        figure = go.Figure(go.Heatmap(
            z=scores.to_numpy(), x=scores.columns, y=scores.index,
            zmin=0, zmax=100,
            colorscale=[[0, "#f3e5d3"], [0.5, "#a9c5ab"], [1, "#176459"]],
            customdata=counts.to_numpy(),
            hovertemplate="%{y}<br>%{x}<br>%{z:.2f}% · %{customdata} tentatives<extra></extra>",
            colorbar=dict(title="%", thickness=10), xgap=4, ygap=4,
        ))
        for category in scores.index:
            for model in scores.columns:
                value = scores.loc[category, model]
                if pd.notna(value):
                    figure.add_annotation(
                        x=model, y=category, text=f"{value:.1f}%", showarrow=False,
                        font=dict(color="#fffefa" if value >= 75 else "#243d35", size=12),
                    )
        finish(figure, max(320, len(scores) * 30 + 90))
        figure.update_yaxes(autorange="reversed")
        figure.update_xaxes(side="top")
        return figure
    figure = px.bar(
        summary, x=dimension, y=metric, color="model_label", barmode="group",
        color_discrete_map=colors, text=metric, labels=LABELS,
        category_orders={"difficulty": ["easy", "medium", "hard"], "question_type": ["boolean", "multiple"]},
        hover_data={"total_questions": True, metric: ":.2f"},
    )
    figure.update_traces(texttemplate="%{y:.1f}%", textposition="outside", cliponaxis=False)
    finish(figure, 400)
    figure.update_yaxes(range=[0, 110], tickvals=[0, 25, 50, 75, 100], ticksuffix=" %")
    return figure


def quality(summary: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for name, values, color in (
        ("Correcte", summary["correct_answers"], "#176459"),
        ("Incorrecte", summary["valid_answers"] - summary["correct_answers"], "#cf7646"),
        ("Invalide", summary["invalid_answers"], "#d4b35c"),
        ("Erreur technique", summary["technical_errors"], "#925773"),
    ):
        figure.add_bar(
            name=name, y=summary["model_label"], x=100 * values / summary["total_questions"],
            customdata=values, orientation="h", marker_color=color,
            hovertemplate=f"%{{y}}<br>{name} : %{{customdata}} (%{{x:.2f}}%)<extra></extra>",
        )
    finish(figure, max(270, len(summary) * 60 + 120))
    figure.update_layout(barmode="stack")
    figure.update_xaxes(range=[0, 100], ticksuffix=" %")
    figure.update_yaxes(showgrid=False)
    return figure
