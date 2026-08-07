from __future__ import annotations

import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.reporting.document import (
    Bullets,
    Document,
    Figure,
    Heading,
    Paragraph,
    Table,
    to_html,
    to_markdown,
)
from nnphysics.reporting.theme import report_stylesheet

TABLE = Table(
    headers=("predictor", "error"),
    rows=(("reference", "0"), ("persistence", "0.4")),
    caption="One number each.",
)


class TestValidation:
    def test_a_table_row_must_match_its_headings(self) -> None:
        with pytest.raises(ValidationError, match="expected 2"):
            Table(headers=("a", "b"), rows=(("only one",),))

    def test_a_table_must_have_a_column(self) -> None:
        with pytest.raises(ValidationError, match="at least one column"):
            Table(headers=(), rows=())

    def test_a_heading_level_must_exist_in_both_formats(self) -> None:
        with pytest.raises(ValidationError, match="1 to 6"):
            Heading("too deep", level=7)


class TestMarkdown:
    def test_the_title_is_the_top_heading(self) -> None:
        assert to_markdown(Document("Run one")).startswith("# Run one")

    def test_it_ends_in_a_single_newline(self) -> None:
        text = to_markdown(Document("Run one", (Paragraph("Done."),)))

        assert text.endswith("Done.\n")
        assert not text.endswith("\n\n")

    def test_a_table_is_a_pipe_table_under_its_caption(self) -> None:
        text = to_markdown(Document("t", (TABLE,)))

        assert "One number each." in text
        assert "| predictor   | error |" in text
        assert "| reference   | 0     |" in text

    def test_bullets_and_headings_render(self) -> None:
        text = to_markdown(Document("t", (Heading("Setup", level=3), Bullets(("one", "two")))))

        assert "### Setup" in text
        assert "- one\n- two" in text

    def test_a_figure_carries_its_caption(self) -> None:
        text = to_markdown(Document("t", (Figure("plots/a.png", "curve", "How to read it."),)))

        assert "![curve](plots/a.png)" in text
        assert "How to read it." in text

    def test_the_same_document_renders_identically_twice(self) -> None:
        document = Document("t", (Paragraph("p"), TABLE))

        assert to_markdown(document) == to_markdown(document)


class TestHtml:
    def test_it_is_a_whole_document(self) -> None:
        text = to_html(Document("Run one"))

        assert text.startswith("<!DOCTYPE html>")
        assert "<title>Run one</title>" in text
        assert text.rstrip().endswith("</html>")

    def test_text_is_escaped(self) -> None:
        text = to_html(Document("t", (Paragraph("1 < 2 & 3 > 2"),)))

        assert "1 &lt; 2 &amp; 3 &gt; 2" in text
        assert "1 < 2" not in text

    def test_a_table_becomes_a_table(self) -> None:
        text = to_html(Document("t", (TABLE,)))

        assert "<caption>One number each.</caption>" in text
        assert "<th>predictor</th>" in text
        assert "<td>persistence</td>" in text

    def test_the_resolver_decides_what_an_image_points_at(self) -> None:
        document = Document("t", (Figure("plots/a.png", "curve"),))

        text = to_html(document, lambda source: f"data:image/png;base64,{len(source)}")

        assert 'src="data:image/png;base64,11"' in text
        assert "plots/a.png" not in text

    def test_the_style_travels_with_the_document(self) -> None:
        """The one stylesheet, inline, so the file needs nothing beside it."""
        text = to_html(Document("t"))

        assert f"<style>\n{report_stylesheet()}</style>" in text
        assert "<link" not in text

    def test_the_same_document_renders_identically_twice(self) -> None:
        document = Document("t", (Paragraph("p"), TABLE))

        assert to_html(document) == to_html(document)


class TestBothFormats:
    def test_they_carry_the_same_numbers(self) -> None:
        document = Document("t", (TABLE,))

        markdown = to_markdown(document)
        html = to_html(document)

        for cell in ("reference", "0.4", "persistence"):
            assert cell in markdown
            assert cell in html
