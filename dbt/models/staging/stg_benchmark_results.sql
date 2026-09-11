select
    benchmark_id,
    question_id,

    category,
    type as question_type,
    difficulty,
    question,
    correct_answer,
    answer_choices,

    model_name,
    prompt_id,
    generation_temperature,
    generation_seed,

    correct_choice,
    ai_answer,
    ai_answer_raw,
    ai_correct,

    response_time,
    status,
    error_message,

    prompt_tokens,
    completion_tokens,
    total_tokens,

    created_at

from {{ source('silver', 'benchmark_results') }}
