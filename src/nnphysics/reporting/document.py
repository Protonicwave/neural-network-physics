"""A report as blocks, before it is a file.

Markdown and HTML are two serialisations of the same document, and the document is built
once. That is what keeps the two reports saying the same thing: there is no second code
path that could drift, only a second way of writing the same blocks down.

The blocks are deliberately few. A report is headings, sentences, tables and figures, and
a richer document model would be a small markup language with all the ways of going wrong
that implies.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from nnphysics.core.errors import ValidationError

__all__ = [
    "Block",
    "Bullets",
    "Document",
    "Figure",
    "Heading",
    "Paragraph",
    "Table",
    "to_html",
    "to_markdown",
]

_MAX_HEADING = 6


@dataclass(frozen=True, slots=True)
class Heading:
    """A section heading.

    Attributes:
        text: The heading.
        level: One to six, as in both formats.
    """

    text: str
    level: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.level <= _MAX_HEADING:
            raise ValidationError(f"a heading level must be 1 to {_MAX_HEADING}, got {self.level}")


@dataclass(frozen=True, slots=True)
class Paragraph:
    """A run of prose.

    Attributes:
        text: The sentence or sentences.
    """

    text: str


@dataclass(frozen=True, slots=True)
class Bullets:
    """An unordered list.

    Attributes:
        items: One string per bullet.
    """

    items: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Table:
    """A table of already formatted cells.

    Formatting happens before a table is built, not while it is rendered, so that both
    serialisations show the same digits.

    Attributes:
        headers: Column headings.
        rows: Cells, one tuple per row, each as long as the headings.
        caption: What the table shows. Rendered above it in both formats.
    """

    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    caption: str = ""

    def __post_init__(self) -> None:
        if not self.headers:
            raise ValidationError("a table must have at least one column")
        for position, row in enumerate(self.rows):
            if len(row) != len(self.headers):
                raise ValidationError(
                    f"table row {position} has {len(row)} cells, expected {len(self.headers)}"
                )


@dataclass(frozen=True, slots=True)
class Figure:
    """An image with a caption.

    Attributes:
        source: Path to the image, relative to the report. The HTML serialisation
            replaces it with whatever the resolver returns, which is how a self contained
            file gets its images.
        alt: Text for a reader who cannot see the image.
        caption: How to read the figure.
    """

    source: str
    alt: str
    caption: str = ""


type Block = Heading | Paragraph | Bullets | Table | Figure


@dataclass(frozen=True, slots=True)
class Document:
    """A whole report.

    Attributes:
        title: Document title, rendered as the top heading and as the HTML title.
        blocks: Everything below it, in order.
    """

    title: str
    blocks: tuple[Block, ...] = field(default_factory=tuple)


def to_markdown(document: Document) -> str:
    """Serialise a document as Markdown.

    Args:
        document: The document.

    Returns:
        The Markdown, ending in a single newline.
    """
    parts = [f"# {document.title}"]
    for block in document.blocks:
        parts.append(_markdown_block(block))
    return "\n\n".join(parts) + "\n"


def to_html(document: Document, resolve: Callable[[str], str] | None = None) -> str:
    """Serialise a document as a single HTML file.

    Args:
        document: The document.
        resolve: Turns a figure's source into something the browser can load. Pass an
            encoder to embed images; the default leaves the path alone, which is only
            useful when the images sit beside the file.

    Returns:
        The HTML, ending in a single newline. It references nothing outside itself as
        long as the resolver does not introduce a reference.
    """
    source = resolve or (lambda path: path)
    body = "\n".join(_html_block(block, source) for block in document.blocks)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(document.title)}</title>\n"
        f"<style>\n{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{html.escape(document.title)}</h1>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def _markdown_block(block: Block) -> str:
    """One block as Markdown."""
    match block:
        case Heading(text=text, level=level):
            return f"{'#' * level} {text}"
        case Paragraph(text=text):
            return text
        case Bullets(items=items):
            return "\n".join(f"- {item}" for item in items)
        case Table():
            return _markdown_table(block)
        case Figure(source=source, alt=alt, caption=caption):
            image = f"![{alt}]({source})"
            return f"{image}\n\n{caption}" if caption else image


def _markdown_table(table: Table) -> str:
    """A table as a pipe table, with its caption above it."""
    widths = [
        max(len(header), *(len(row[column]) for row in table.rows)) if table.rows else len(header)
        for column, header in enumerate(table.headers)
    ]
    lines = [
        _markdown_row(table.headers, widths),
        _markdown_row(["-" * width for width in widths], widths),
    ]
    lines += [_markdown_row(row, widths) for row in table.rows]
    body = "\n".join(lines)
    return f"{table.caption}\n\n{body}" if table.caption else body


def _markdown_row(cells: Sequence[str], widths: Sequence[int]) -> str:
    """One row, padded so the source is readable as well as the rendering."""
    padded = " | ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
    return f"| {padded} |"


def _html_block(block: Block, resolve: Callable[[str], str]) -> str:
    """One block as HTML."""
    match block:
        case Heading(text=text, level=level):
            return f"<h{level}>{html.escape(text)}</h{level}>"
        case Paragraph(text=text):
            return f"<p>{html.escape(text)}</p>"
        case Bullets(items=items):
            entries = "".join(f"<li>{html.escape(item)}</li>" for item in items)
            return f"<ul>{entries}</ul>"
        case Table():
            return _html_table(block)
        case Figure(source=source, alt=alt, caption=caption):
            attributes = (
                f'src="{html.escape(resolve(source), quote=True)}" '
                f'alt="{html.escape(alt, quote=True)}"'
            )
            image = f"<img {attributes}>"
            body = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
            return f"<figure>{image}{body}</figure>"


def _html_table(table: Table) -> str:
    """A table as HTML, with its caption as a table caption."""
    caption = f"<caption>{html.escape(table.caption)}</caption>" if table.caption else ""
    head = "".join(f"<th>{html.escape(header)}</th>" for header in table.headers)
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in table.rows
    )
    return f"<table>{caption}<thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


_CSS = """\
:root { color-scheme: light dark; }
body {
  margin: 0 auto; padding: 2rem 1.5rem; max-width: 60rem;
  font-family: system-ui, sans-serif; line-height: 1.5;
}
h1, h2, h3 { line-height: 1.25; }
h2 {
  margin-top: 2.5rem; padding-bottom: 0.2rem;
  border-bottom: 1px solid rgba(127, 127, 127, 0.35);
}
table { border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; width: 100%; }
caption { text-align: left; padding-bottom: 0.4rem; font-style: italic; }
th, td {
  border-bottom: 1px solid rgba(127, 127, 127, 0.35);
  padding: 0.3rem 0.6rem; text-align: left;
}
th { font-weight: 600; }
td:not(:first-child), th:not(:first-child) {
  text-align: right; font-variant-numeric: tabular-nums;
}
figure { margin: 1.5rem 0; }
img { max-width: 100%; height: auto; }
figcaption { font-size: 0.9rem; font-style: italic; padding-top: 0.4rem; }
"""
