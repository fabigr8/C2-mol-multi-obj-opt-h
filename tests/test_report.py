import pytest

from rhmoo.report import FIGURE_NAMES, generate_results_md

pytestmark = pytest.mark.smoke


def test_generate_results_md_contains_every_section(results_dir):
    content = generate_results_md(results_dir).read_text()

    assert content.startswith("# RESULTS")
    for heading in (
        "## Per-arm summary",
        "## H3: objective-vs-independent divergence -- headline scorer variant",
        "## H3: objective-vs-independent divergence -- all scorer variants",
        "## H3b: decoupling generation (primary pre-declared margin)",
        "## Exploit clusters (optimized arms)",
    ):
        assert heading in content


def test_results_md_contains_no_prose(results_dir):
    """brief SS5: RESULTS.md is a machine-generated artifact -- numbers, tables
    and figure embeds only, no conclusions, narrative or claims. Enforced
    structurally so prose cannot slip in undetected.
    """
    content = generate_results_md(results_dir).read_text()
    offending = [
        line
        for line in content.splitlines()
        if line.strip() and not (line.startswith("#") or line.startswith("|") or line.startswith("!["))
    ]
    assert offending == []


def test_figures_are_embedded_when_present(results_dir):
    figures = results_dir / "figures"
    figures.mkdir()
    for name in FIGURE_NAMES:
        (figures / f"{name}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    content = generate_results_md(results_dir).read_text()
    for name in FIGURE_NAMES:
        assert f"![{name}](figures/{name}.png)" in content


def test_missing_figures_are_skipped_silently(results_dir):
    content = generate_results_md(results_dir).read_text()
    assert "![" not in content
