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
        '<header class="masthead"><div class="wrap">nnphysics</div></header>\n'
        '<main class="wrap">\n'
        f"<h1>{html.escape(document.title)}</h1>\n"
        f"{body}\n"
        "</main>\n"
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
:root {
  color-scheme: light dark;
  --plane: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --rule: rgba(11, 11, 11, 0.10);
  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --sans: ui-sans-serif, system-ui, "Segoe UI", Inter, sans-serif;
  --mono: ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --plane: #0d0d0d;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --rule: rgba(255, 255, 255, 0.10);
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--plane); color: var(--ink);
  font-family: var(--sans); font-size: 16px; line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 62rem; margin: 0 auto; padding: 0 1.5rem; }
.masthead {
  border-bottom: 1px solid var(--rule); margin-bottom: 3.5rem;
  font-family: var(--mono); font-size: 0.78rem; color: var(--muted);
}
.masthead .wrap { padding-top: 0.9rem; padding-bottom: 0.9rem; }
main { padding-bottom: 6rem; }
h1, h2, h3, h4, h5, h6 {
  font-family: var(--serif); font-weight: 600; line-height: 1.2;
  letter-spacing: -0.01em; color: var(--ink);
}
h1 { font-size: clamp(1.9rem, 4vw, 2.6rem); margin: 0 0 1.5rem; max-width: 24ch; }
h2 {
  font-size: clamp(1.3rem, 2.6vw, 1.6rem); margin: 3.5rem 0 1rem;
  padding-top: 1.75rem; border-top: 1px solid var(--rule);
}
h3 { font-size: 1.1rem; margin: 2.25rem 0 0.6rem; }
p, li { color: var(--ink-2); }
p { max-width: 38rem; }
ul { max-width: 38rem; padding-left: 1.1rem; }
li { margin-bottom: 0.3rem; }
code { font-family: var(--mono); font-size: 0.86em; }
table {
  border-collapse: collapse; width: 100%; margin: 1.75rem 0;
  font-size: 0.88rem; display: block; overflow-x: auto;
}
caption {
  text-align: left; padding-bottom: 0.6rem;
  font-size: 0.86rem; color: var(--muted);
}
th, td { border-bottom: 1px solid var(--rule); padding: 0.55rem 0.8rem; text-align: left; }
th {
  font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--muted); white-space: nowrap;
}
td { color: var(--ink-2); }
td:first-child { color: var(--ink); }
td:not(:first-child), th:not(:first-child) {
  text-align: right; font-variant-numeric: tabular-nums; font-family: var(--mono);
}
th:not(:first-child) { font-family: var(--sans); }
tbody tr:hover { background: var(--surface); }
figure { margin: 2rem 0; }
figure img {
  display: block; width: 100%; height: auto;
  background: #ffffff; border: 1px solid var(--rule);
  border-radius: 8px; padding: 0.75rem;
}
figcaption { font-size: 0.86rem; color: var(--muted); padding-top: 0.9rem; max-width: 44rem; }
"""
