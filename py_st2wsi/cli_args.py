"""CLI argument parsing for ST2WSI.

The Java plugin's ``parseArg`` splits a single argument string on
whitespace *at bracket/quote depth zero* and trims a matching outer
pair of ``[]`` or ``''``/``""`` off each value. The shell wrapper
relies on this by passing ``tgtStain=[H&E 2]`` to quote a stain name
that contains a space. Any port has to honour the same rule or it
will silently mis-parse H&E 2.

The :func:`parse_arg` function below is a faithful Python re-implementation
of the Java method (single-file, no external deps) and accepts the
same input shape: a single string ``"key=value key=value …"``.
"""
from __future__ import annotations

from typing import Dict, List


def parse_arg(arg: str) -> Dict[str, str]:
    """Parse a Java-ImageJ style ``key=value`` argument string.

    Mirrors ``ST2WSI_Registration.parseArg`` line-by-line.

    * Splits on whitespace and commas at bracket/quote depth zero.
    * Accepts ``[bracket]`` and ``'single'``/``"double"`` quoted values;
      a leading/trailing matching pair is stripped.
    * A pair that doesn't contain ``=`` is silently dropped
      (matches the Java side, which only stores pairs where
      ``0 < idx < len - 1``).
    """
    out: Dict[str, str] = {}
    if not arg or not arg.strip():
        return out

    pairs: List[str] = []
    cur: List[str] = []
    quote = ""
    depth = 0
    for c in arg.strip():
        if quote:
            if c == quote:
                quote = ""
            cur.append(c)
        elif c in ("'", '"'):
            quote = c
            cur.append(c)
        elif c == "[":
            depth += 1
            cur.append(c)
        elif c == "]":
            if depth > 0:
                depth -= 1
            cur.append(c)
        elif depth == 0 and (c.isspace() or c == ","):
            if cur:
                pairs.append("".join(cur))
                cur = []
        else:
            cur.append(c)
    if cur:
        pairs.append("".join(cur))

    for pair in pairs:
        idx = pair.find("=")
        if 0 < idx < len(pair) - 1:
            key = pair[:idx].strip()
            value = pair[idx + 1:].strip()
            if len(value) >= 2:
                first, last = value[0], value[-1]
                if (first == "[" and last == "]") or (
                    (first == "'" and last == "'")
                    or (first == '"' and last == '"')
                ):
                    value = value[1:-1]
            out[key] = value
    return out


def to_java_arg_string(params: Dict[str, str]) -> str:
    """Round-trip helper used by tests and the wrapper script.

    Emits the same shape the shell wrapper sends to Fiji::

        key=value key[spaced value]=value
    """
    parts = []
    for k, v in params.items():
        if " " in v or "\t" in v:
            parts.append(f"{k}=[{v}]")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)
