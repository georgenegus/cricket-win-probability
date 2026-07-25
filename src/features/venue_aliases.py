"""
Venue normalization and alias resolution layer.
Maps external, sponsor-branded, or abbreviated venue strings to canonical historical venue keys.
"""

import re
from difflib import SequenceMatcher

# Explicit Canonical Alias Mappings
VENUE_ALIASES = {
    # English County Cricket Grounds
    "the oval": "kennington oval, london",
    "oval": "kennington oval, london",
    "the oval, london": "kennington oval, london",
    "kennington oval": "kennington oval, london",
    "emirates old trafford": "old trafford, manchester",
    "old trafford": "old trafford, manchester",
    "old trafford, manchester": "old trafford, manchester",
    "lord's": "lord's, london",
    "lords": "lord's, london",
    "lord's, london": "lord's, london",
    "lord's cricket ground": "lord's, london",
    "lords cricket ground": "lord's, london",
    "trent bridge": "trent bridge, nottingham",
    "trent bridge, nottingham": "trent bridge, nottingham",
    "headingley": "headingley, leeds",
    "headingley, leeds": "headingley, leeds",
    "headingley stadium": "headingley, leeds",
    "edgbaston": "edgbaston, birmingham",
    "edgbaston, birmingham": "edgbaston, birmingham",
    "the cooper associates county ground": "the cooper associates county ground, taunton",
    "the cooper associates county ground, taunton": "the cooper associates county ground, taunton",
    "taunton": "the cooper associates county ground, taunton",
    "county ground, taunton": "the cooper associates county ground, taunton",
    "the royal & sun alliance county ground": "county ground, bristol",
    "seat unique stadium": "county ground, bristol",
    "bristol": "county ground, bristol",
    "county ground, bristol": "county ground, bristol",
    "cloud county ground": "county ground, chelmsford",
    "chelmsford": "county ground, chelmsford",
    "county ground, chelmsford": "county ground, chelmsford",
    "incora county ground": "county ground, derby",
    "derby": "county ground, derby",
    "county ground, derby": "county ground, derby",
    "1st central county ground": "county ground, hove",
    "hove": "county ground, hove",
    "county ground, hove": "county ground, hove",
    "county ground, new road": "county ground, new road, worcester",
    "new road": "county ground, new road, worcester",
    "worcester": "county ground, new road, worcester",
    "county ground, worcester": "county ground, new road, worcester",
    "county ground, northampton": "county ground, northampton",
    "northampton": "county ground, northampton",
    "wantage road": "county ground, northampton",
    "uptonsteel county ground": "grace road, leicester",
    "grace road": "grace road, leicester",
    "leicester": "grace road, leicester",
    "spitfire ground": "st lawrence ground, canterbury",
    "canterbury": "st lawrence ground, canterbury",
    "st lawrence ground": "st lawrence ground, canterbury",
    "utilita bowl": "the rose bowl, southampton",
    "ageas bowl": "the rose bowl, southampton",
    "rose bowl": "the rose bowl, southampton",
    "the rose bowl": "the rose bowl, southampton",
    "southampton": "the rose bowl, southampton",
    "sophia gardens": "sophia gardens, cardiff",
    "cardiff": "sophia gardens, cardiff",

    # IPL / Indian Stadiums
    "chinnaswamy stadium": "m chinnaswamy stadium, bengaluru",
    "m. chinnaswamy stadium": "m chinnaswamy stadium, bengaluru",
    "mchinnaswamy stadium": "m chinnaswamy stadium, bengaluru",
    "m chinnaswamy stadium": "m chinnaswamy stadium, bengaluru",
    "m chinnaswamy stadium, bengaluru": "m chinnaswamy stadium, bengaluru",
    "m chinnaswamy stadium, bangalore": "m chinnaswamy stadium, bengaluru",
    "bengaluru": "m chinnaswamy stadium, bengaluru",
    "chepauk": "ma chidambaram stadium, chepauk, chennai",
    "ma chidambaram stadium": "ma chidambaram stadium, chepauk, chennai",
    "ma chidambaram stadium, chepauk": "ma chidambaram stadium, chepauk, chennai",
    "chennai": "ma chidambaram stadium, chepauk, chennai",
    "wankhede": "wankhede stadium, mumbai",
    "wankhede stadium": "wankhede stadium, mumbai",
    "wankhede stadium, mumbai": "wankhede stadium, mumbai",
    "mumbai": "wankhede stadium, mumbai",
    "eden gardens": "eden gardens, kolkata",
    "eden gardens, kolkata": "eden gardens, kolkata",
    "kolkata": "eden gardens, kolkata",
    "narendra modi stadium": "narendra modi stadium, ahmedabad",
    "narendra modi stadium, ahmedabad": "narendra modi stadium, ahmedabad",
    "ahmedabad": "narendra modi stadium, ahmedabad",
    "arun jaitley stadium": "arun jaitley stadium, delhi",
    "arun jaitley stadium, delhi": "arun jaitley stadium, delhi",
    "delhi": "arun jaitley stadium, delhi",
    "feroz shah kotla": "arun jaitley stadium, delhi",
    "rajiv gandhi international stadium": "rajiv gandhi international stadium, uppal, hyderabad",
    "rajiv gandhi international stadium, uppal": "rajiv gandhi international stadium, uppal, hyderabad",
    "hyderabad": "rajiv gandhi international stadium, uppal, hyderabad",
    "sawai mansingh stadium": "sawai mansingh stadium, jaipur",
    "sawai mansingh stadium, jaipur": "sawai mansingh stadium, jaipur",
    "jaipur": "sawai mansingh stadium, jaipur",
    "ekana cricket stadium": "bharat ratna shri atal bihari vajpayee ekana cricket stadium, lucknow",
    "lucknow": "bharat ratna shri atal bihari vajpayee ekana cricket stadium, lucknow",
    "is bindra stadium": "punjab cricket association is bindra stadium, mohali, chandigarh",
    "mohali": "punjab cricket association is bindra stadium, mohali, chandigarh",

    # International Stadiums
    "mcg": "melbourne cricket ground",
    "melbourne": "melbourne cricket ground",
    "scg": "sydney cricket ground",
    "sydney": "sydney cricket ground",
    "gabba": "brisbane cricket ground, woolloongabba, brisbane",
    "brisbane": "brisbane cricket ground, woolloongabba, brisbane",
    "optus stadium": "perth stadium",
    "perth": "perth stadium",
    "adelaide oval": "adelaide oval",
    "adelaide": "adelaide oval",
}

# Ambiguous venue names that should NEVER be bound via fuzzy/substring search alone
AMBIGUOUS_VENUES = {
    "county ground",
    "national stadium",
    "sports ground",
    "cricket ground",
    "stadium",
    "oval",
    "city oval",
}


def clean_venue_string(venue_name: str) -> str:
    """Standardizes string casing, punctuation, and extra whitespace."""
    if not venue_name or str(venue_name).strip().upper() in ["N/A", "NONE", "UNKNOWN", ""]:
        return ""
    text = str(venue_name).lower().strip()
    text = re.sub(r"[^\w\s,\']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# VENUE_ALIASES keys were hand-typed and are not guaranteed to already be in
# clean_venue_string's canonical form (e.g. inconsistent punctuation/spacing
# across entries). Normalizing the keys once at import time means a lookup
# only ever has to clean the *input* string, and typos/spacing differences in
# the hardcoded keys above can't silently create a second, unmatched entry.
_VENUE_ALIASES_NORMALIZED = {clean_venue_string(k): v for k, v in VENUE_ALIASES.items()}


def canonicalize_for_aggregation(raw_venue_name: str) -> str:
    """
    Deterministic canonicalization used at aggregation time (VenueEngine), where
    there is no pre-existing "available venues" set to fuzzy-match against yet.

    Cleans punctuation/casing the same way as resolve_venue_alias, then applies
    the static VENUE_ALIASES map if the cleaned string is a known variant. This
    merges near-duplicate raw venue strings (e.g. "M.Chinnaswamy Stadium",
    "M Chinnaswamy Stadium, Bangalore", "M Chinnaswamy Stadium, Bengaluru") into
    one canonical bucket at the point historical stats are aggregated, instead
    of letting each raw spelling silently split a venue's history across
    several rows (each with fewer matches and a less reliable average).

    Only the static alias map is used here (no fuzzy/substring matching) —
    aggregation runs once over the full historical dataset, so silently
    merging two genuinely different grounds via a fuzzy guess would corrupt
    training data. Fuzzy resolution is reserved for `resolve_venue_alias`,
    which runs against a known, already-deduplicated set of venues.
    """
    cleaned = clean_venue_string(raw_venue_name)
    if not cleaned:
        return ""
    return _VENUE_ALIASES_NORMALIZED.get(cleaned, cleaned)


def resolve_venue_alias(raw_venue_name: str, available_venues: dict) -> str | None:
    """
    Resolves a raw venue string to a canonical historical venue key in available_venues.

    Resolution order:
    1. Alias-map hit -> canonical key -> exact dictionary lookup
    2. Exact normalized match
    3. High-confidence token-overlap match (preventing ambiguous "County Ground" collisions)

    Returns canonical venue key string, or None if unknown/ambiguous.
    """
    cleaned = clean_venue_string(raw_venue_name)
    if not cleaned or cleaned in AMBIGUOUS_VENUES:
        return None

    # Step 1: Explicit Alias Map Lookup
    if cleaned in _VENUE_ALIASES_NORMALIZED:
        canonical = _VENUE_ALIASES_NORMALIZED[cleaned]
        if canonical in available_venues:
            return canonical

    # Step 2: Exact Normalized Match
    if cleaned in available_venues:
        return cleaned

    for v_key in available_venues:
        if clean_venue_string(v_key) == cleaned:
            return v_key

    # Step 3: High-Confidence Token / Substring Matching (With Ambiguity Guard)
    candidate_matches = []
    cleaned_tokens = set(cleaned.split())

    for v_key in available_venues:
        clean_key = clean_venue_string(v_key)
        if not clean_key:
            continue

        if clean_key in AMBIGUOUS_VENUES:
            continue

        key_tokens = set(clean_key.split())
        token_overlap = len(cleaned_tokens.intersection(key_tokens))
        ratio = SequenceMatcher(None, cleaned, clean_key).ratio()

        is_sub = (cleaned in clean_key or clean_key in cleaned) and len(cleaned_tokens) >= 2

        if ratio >= 0.82 or is_sub or (token_overlap >= 2 and ratio >= 0.70):
            candidate_matches.append((v_key, ratio))

    if not candidate_matches:
        return None

    candidate_matches.sort(key=lambda x: x[1], reverse=True)

    # Ambiguity check: if top 2 candidates have very close scores, reject as ambiguous
    if len(candidate_matches) > 1:
        top_score = candidate_matches[0][1]
        runner_up_score = candidate_matches[1][1]
        if abs(top_score - runner_up_score) < 0.05:
            return None

    return candidate_matches[0][0]
