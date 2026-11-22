"""Small HTML-to-text conversion using Lexbor through selectolax."""

from __future__ import annotations

from selectolax.parser import HTMLParser


def html_to_text(source: str) -> tuple[str, str]:
    tree = HTMLParser(source)
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node is not None else ""
    for selector in ("script", "style", "noscript", "svg", "nav", "footer"):
        for node in tree.css(selector):
            node.decompose()
    root = tree.body or tree.root
    text = root.text(separator="\n", strip=True) if root is not None else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return title, "\n\n".join(lines)
