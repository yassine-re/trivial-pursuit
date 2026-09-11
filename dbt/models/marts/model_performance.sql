select
    model_name,
    prompt_id,
    generation_temperature,
    generation_seed,

    count(*) as total_questions,

    sum(
        case
            when ai_correct then 1
            else 0
        end
    ) as correct_answers,

    sum(
        case
            when status = 'success' then 1
            else 0
        end
    ) as valid_answers,

    sum(
        case
            when status = 'invalid_answer' then 1
            else 0
        end
    ) as invalid_answers,

    sum(
        case
            when status = 'error' then 1
            else 0
        end
    ) as technical_errors,

    round(100.0 * correct_answers / nullif(total_questions, 0), 2)
        as overall_accuracy,

    round(100.0 * correct_answers / nullif(valid_answers, 0), 2)
        as valid_answer_accuracy,

    round(100.0 * valid_answers / nullif(total_questions, 0), 2)
        as valid_answer_rate,

    round(100.0 * invalid_answers / nullif(total_questions, 0), 2)
        as invalid_answer_rate,

    round(100.0 * technical_errors / nullif(total_questions, 0), 2)
        as technical_error_rate,

    round(
        avg(response_time),
        3
    ) as avg_response_time,

    round(
        avg(prompt_tokens),
        2
    ) as avg_prompt_tokens,

    round(
        avg(completion_tokens),
        2
    ) as avg_completion_tokens

from {{ ref('stg_benchmark_results') }}

group by
    model_name,
    prompt_id,
    generation_temperature,
    generation_seed
