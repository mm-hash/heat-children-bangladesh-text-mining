"""Create primary public-facing and optional scientific comparison corpora.

Input:  heatwave_corpus_cleaned.csv
Outputs:
  heatwave_public_facing_primary.csv
  heatwave_scientific_extension.csv
  heatwave_corpus_scope_audit.csv

This script never deletes documents. It adds explicit scope fields and writes
separate analysis files so the scientific documents remain available for a
later comparison.
"""

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


INPUT = Path("heatwave_corpus_cleaned.csv")
PRIMARY_OUTPUT = Path("heatwave_public_facing_primary.csv")
SCIENTIFIC_OUTPUT = Path("heatwave_scientific_extension.csv")
AUDIT_OUTPUT = Path("heatwave_corpus_scope_audit.csv")


def as_bool(series: pd.Series) -> pd.Series:
    """Convert common spreadsheet truth values safely to Boolean."""
    return series.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def normalize_source(value: object) -> str:
    value = str(value).strip().lower()
    mapping = {
        "news paper": "newspaper",
        "newspaper": "newspaper",
        "civil society": "civil_society",
        "scientific": "scientific",
        "others": "other_public_facing",
        "other": "other_public_facing",
    }
    return mapping.get(value, value.replace(" ", "_"))


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    text = text.lower()
    return any(term in text for term in terms)


def main() -> None:
    df = pd.read_csv(INPUT)

    required = {
        "doc_id",
        "url",
        "source_type",
        "clean_text",
        "include_in_clean_analysis",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["source_group"] = df["source_type"].map(normalize_source)
    df["domain"] = df["url"].fillna("").map(
        lambda value: urlparse(str(value)).netloc.lower().removeprefix("www.")
    )
    df["previously_analysis_ready"] = as_bool(df["include_in_clean_analysis"])

    # The National Hospital webpage (DOC_057) is public-facing health content.
    # All scientific/academic items are retained, but moved out of the primary
    # corpus to prevent full journal articles from being mixed with public-facing
    # news and civil-society documents.
    public_groups = {"newspaper", "civil_society", "other_public_facing"}
    df["audience_scope"] = "requires_manual_review"
    df.loc[df["source_group"].isin(public_groups), "audience_scope"] = (
        "public_facing"
    )
    df.loc[df["source_group"].eq("scientific"), "audience_scope"] = (
        "scientific_or_academic"
    )

    # Confirmed after reading the cleaned texts: these discuss heat exposure in
    # Bangladesh generally but do not discuss children or child-related outcomes.
    child_scope_exclusions = {"DOC_073", "DOC_077"}

    df["include_primary"] = (
        df["previously_analysis_ready"]
        & df["audience_scope"].eq("public_facing")
        & ~df["doc_id"].isin(child_scope_exclusions)
    )
    df["include_scientific_extension"] = (
        df["previously_analysis_ready"]
        & df["audience_scope"].eq("scientific_or_academic")
    )

    df["primary_scope_reason"] = ""
    df.loc[df["include_primary"], "primary_scope_reason"] = (
        "Included: analysis-ready public-facing online content"
    )
    df.loc[df["include_scientific_extension"], "primary_scope_reason"] = (
        "Excluded from primary corpus; retained for scientific comparison"
    )
    df.loc[df["doc_id"].isin(child_scope_exclusions), "primary_scope_reason"] = (
        "Excluded from child-focused corpus: no child population or outcome"
    )
    df.loc[~df["previously_analysis_ready"], "primary_scope_reason"] = (
        "Excluded during previous cleaning or QA"
    )

    # These are review flags, not automatic exclusions. They help verify that
    # the changed research question is still supported by every primary item.
    text = df["clean_text"].fillna("").astype(str)
    df["mentions_child_terms"] = text.map(
        lambda value: contains_any(
            value,
            (
                "child", "children", "infant", "baby", "babies",
                "newborn", "adolescent", "student", "school", "maternal",
                "pregnan",
            ),
        )
    )
    df["mentions_heat_terms"] = text.map(
        lambda value: contains_any(
            value,
            (
                "heat", "temperature", "hot weather", "thermal",
                "heatwave", "heat wave",
            ),
        )
    )
    df["scope_review_needed"] = df["include_primary"] & (
        ~df["mentions_child_terms"] | ~df["mentions_heat_terms"]
    )

    primary = df.loc[df["include_primary"]].copy()
    scientific = df.loc[df["include_scientific_extension"]].copy()

    if primary["doc_id"].duplicated().any():
        raise ValueError("Duplicate doc_id found in primary corpus")
    if scientific["doc_id"].duplicated().any():
        raise ValueError("Duplicate doc_id found in scientific corpus")
    if set(primary["doc_id"]) & set(scientific["doc_id"]):
        raise ValueError("A document appears in both analysis corpora")
    if primary["clean_text"].fillna("").str.strip().eq("").any():
        raise ValueError("Primary corpus contains blank clean_text")

    primary.to_csv(PRIMARY_OUTPUT, index=False)
    scientific.to_csv(SCIENTIFIC_OUTPUT, index=False)
    df.to_csv(AUDIT_OUTPUT, index=False)

    print(f"Primary public-facing corpus: {len(primary)} documents")
    print(primary["source_group"].value_counts().to_string())
    print(f"\nScientific extension corpus: {len(scientific)} documents")
    print(f"Primary documents flagged for manual scope review: "
          f"{int(primary['scope_review_needed'].sum())}")
    print(f"\nWrote: {PRIMARY_OUTPUT}")
    print(f"Wrote: {SCIENTIFIC_OUTPUT}")
    print(f"Wrote: {AUDIT_OUTPUT}")


if __name__ == "__main__":
    main()
