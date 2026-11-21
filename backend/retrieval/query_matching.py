import re


SECTION_QUERY_PATTERNS = {
    "Frontend": re.compile(r"(?<!\w)front(?:end|-end)(?!\w)", re.IGNORECASE),
    "Backend": re.compile(r"(?<!\w)back(?:end|-end)(?!\w)", re.IGNORECASE),
    "Databases & Storage": re.compile(
        r"(?<!\w)(?:database|databases|storage)(?!\w)",
        re.IGNORECASE,
    ),
    "Infrastructure & DevOps": re.compile(
        r"(?<!\w)(?:infrastructure|devops)(?!\w)",
        re.IGNORECASE,
    ),
}


def sections_in_query(query: str) -> set[str]:
    return {
        section
        for section, pattern in SECTION_QUERY_PATTERNS.items()
        if pattern.search(query)
    }


def exact_search_term(query: str) -> str | None:
    stripped_query = query.strip()
    if re.fullmatch(r"[\w-]+", stripped_query):
        return stripped_query
    return None


def contains_exact_term(text: str, term: str) -> bool:
    return re.search(
        rf"(?<!\w){re.escape(term)}(?!\w)",
        text,
        flags=re.IGNORECASE,
    ) is not None
