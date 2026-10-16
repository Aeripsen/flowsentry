"""The citation metadata agrees with itself: CITATION.cff, pyproject.toml and the
README name the same version, paper DOI and dataset page, so a reviewer who copies
any one of them gets the same reference."""
from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DOI = "10.5220/0015033600004103"


def _cff() -> dict:
    return yaml.safe_load((REPO / "CITATION.cff").read_text(encoding="utf-8"))


def test_cff_version_matches_the_package():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert str(_cff()["version"]) == project["version"]


def test_cff_cites_the_published_paper():
    paper = _cff()["preferred-citation"]
    assert paper["doi"] == DOI
    assert paper["type"] == "conference-paper"
    assert (paper["start"], paper["end"], paper["year"]) == (49, 68, 2026)
    assert paper["authors"][0]["family-names"] == "Jafari"
    assert "Cloud-Generated" in " ".join(paper["title"].split())


def test_readme_carries_the_same_doi_and_dataset_page():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    dataset = _cff()["references"][0]
    assert f"https://doi.org/{DOI}" in readme
    assert f"doi       = {{{DOI}}}" in readme
    assert dataset["url"] in readme
    assert dataset["license"] == "CC-BY-4.0"
    assert "accepted SECRYPT" not in readme  # published since July 2026
