from backend.retrieval.query_matching import (
    contains_exact_term,
    exact_search_term,
    sections_in_query,
)


def test_single_word_query_is_exact_search():
    assert exact_search_term("  Frontend  ") == "Frontend"


def test_question_uses_semantic_search():
    assert exact_search_term("What does the frontend use?") is None


def test_exact_term_matching_ignores_case():
    assert contains_exact_term("Frontend uses Next.js.", "frontend")


def test_exact_term_matching_does_not_match_part_of_a_word():
    assert not contains_exact_term("The frontends use Next.js.", "frontend")


def test_query_selects_document_sections():
    assert sections_in_query("show frontend") == {"Frontend"}
    assert sections_in_query("frontend and backend relationship") == {
        "Frontend",
        "Backend",
    }
    assert sections_in_query("which database stores playlists") == {
        "Databases & Storage"
    }
