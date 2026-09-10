"""Prompts versionnés utilisés par le benchmark."""

from __future__ import annotations

import string


PROMPT_TEMPLATES = {
    "letter_only_v1": (
        "You are answering a general knowledge multiple-choice benchmark.\n"
        "Choose exactly one option. Reply only with its letter, without any "
        "explanation.\n\n"
        "Question: {question}\n\n"
        "Options:\n{options}\n\n"
        "Answer:"
    ),
}


def render_prompt(question: str, choices: list[str], prompt_id: str) -> str:
    """Rend le prompt exact envoyé au modèle."""

    try:
        template = PROMPT_TEMPLATES[prompt_id]
    except KeyError as error:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"Prompt inconnu {prompt_id!r}. Prompts disponibles : {available}."
        ) from error
    if not question or not choices or len(choices) > len(string.ascii_uppercase):
        raise ValueError("Question ou choix invalides pour construire le prompt.")
    options = "\n".join(
        f"{string.ascii_uppercase[index]}. {choice}"
        for index, choice in enumerate(choices)
    )
    return template.format(question=question, options=options)
