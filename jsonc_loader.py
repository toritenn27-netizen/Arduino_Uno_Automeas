"""
JSONC loader: JSON with // line comments and /* block comments */ support.
"""
import json
import re


_TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"'   # JSON string literal (preserved as-is)
    r'|/\*.*?\*/'           # block comment
    r'|//[^\n]*',           # line comment
    re.DOTALL,
)


def _strip_comments(m: "re.Match[str]") -> str:
    s = m.group(0)
    return s if s.startswith('"') else ""


def load_jsonc(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = _TOKEN_RE.sub(_strip_comments, text)
    return json.loads(text)
