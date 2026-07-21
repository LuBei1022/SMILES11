from src.metrics.generation.faithfulness import faithfulness
from src.metrics.generation.relevance import answer_relevance


def generation_trace(language="en"):
    if language == "ru":
        query = "Как изменить пароль карты?"
        answer = "Пароль карты можно изменить в приложении."
        context = "В приложении банка можно изменить пароль карты."
    else:
        query = "How can I change my card password?"
        answer = "You can change your card password in the application."
        context = "Customers can change a card password in the application."
    return {
        "trace_id": f"generation-{language}",
        "language": language,
        "query": query,
        "context_construction": {"final_context": context},
        "generation": {"final_answer": answer, "status": "success"},
    }


def test_answer_relevance_scores_supported_english_answer():
    result = answer_relevance(generation_trace("en"), {"relevance_relevant": 0.5})

    assert 0.0 <= result.score <= 1.0
    assert result.label == "relevant"
    assert result.evidence["shared_tokens"]


def test_answer_relevance_handles_russian_text():
    result = answer_relevance(generation_trace("ru"), {"relevance_relevant": 0.5})

    assert result.score >= 0.5
    assert result.label == "relevant"


def test_answer_relevance_skips_empty_answer():
    trace = generation_trace()
    trace["generation"]["final_answer"] = ""

    result = answer_relevance(trace, {})

    assert result.status == "skipped"
    assert result.score is None
    assert result.label == "unknown"


def test_faithfulness_scores_context_supported_sentences():
    result = faithfulness(generation_trace("en"), {"faithfulness_supported": 0.8, "faithfulness_partial": 0.5})

    assert result.score == 1.0
    assert result.label == "supported"
    assert result.evidence["supported_sentence_count"] == 1


def test_faithfulness_skips_without_context():
    trace = generation_trace()
    trace["context_construction"]["final_context"] = ""

    result = faithfulness(trace, {})

    assert result.status == "skipped"
    assert result.score is None
    assert result.label == "unknown"
