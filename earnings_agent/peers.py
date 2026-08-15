"""Peer group resolution.

Declared in config rather than inferred. SEC SIC codes look like the obvious
automatic source and are not usable here: SanDisk is 3572 "Computer Storage
Devices" while Micron is 3674 "Semiconductors & Related Devices", so SIC would
split precisely the two companies whose NAND commentary belongs side by side.

This matters for theses specifically. A commodity or cyclical name is driven by
an industry pricing cycle that shows up in competitors' disclosures before its
own - SanDisk's move is a NAND shortage story, and almost none of that signal
is in SanDisk's own 8-K. Reasoning from a single issuer's filing is how you get
a confident and uninformed answer.
"""

from __future__ import annotations


def resolve(ticker: str, groups: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """Peers for a ticker. Returns (peers, group_names).

    A ticker may sit in several groups (a memory maker is both a storage and a
    semis play); the union is returned, self excluded.
    """
    t = ticker.upper()
    peers: list[str] = []
    names: list[str] = []
    for name, members in groups.items():
        upper = [m.upper() for m in members]
        if t not in upper:
            continue
        names.append(name)
        for m in upper:
            if m != t and m not in peers:
                peers.append(m)
    return peers, names


def describe(ticker: str, groups: dict[str, list[str]]) -> str:
    peers, names = resolve(ticker, groups)
    if not peers:
        return (
            f"{ticker} is not in any peer group. A thesis built from one issuer's "
            f"filing sees only that issuer - add a group to config.toml."
        )
    return f"{ticker} peers via {', '.join(names)}: {', '.join(peers)}"
