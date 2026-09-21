"""One view of a finished run, in exactly three panes.

    1. Pending actions   what the system wants to do but may not do alone
    2. Flagged           what it refused to act on, and what it did instead
    3. Commitments       the dates the inbox commits the owner to

Assembled from what earlier parts already decided, never from a fresh opinion: the
pending list is `gate.screen` applied to the recorded drafts, the flagged list is
what the gate refused plus `hostile.found` plus the drafts that failed their own
checks, and the calendar is `commitments.extract`. No model runs. That is what
makes the page reproducible from a run rather than hand-assembled -- the same
`decisions.json` produces the same page, every time.

Two files come out of it, and the order matters. `state/dashboard.json` is the
data; `dashboard.html` is rendered from that data and holds nothing the JSON does
not. A reader who distrusts the page can read the JSON, and a marking script can
read it without parsing HTML.

Usage:
    python dashboard.py     # build both files from the recorded run
"""

import html
import json
from datetime import datetime, timezone

import actions
import commitments
import config
import gate
import hostile
import mailstore
import rules


def pending(box, rows, folders):
    """Pane 1. Everything the gate would stop and ask a person about.

    Each row is the message, the action proposed, and why it cannot happen alone.
    Refusals are not pending: a refused action is not waiting for anybody, and
    putting it here would tell the owner they have a decision to make when they
    do not.
    """
    found = []
    for proposal in gate.proposals_from_decisions(rows, box):
        refusals, asks = gate.screen(proposal, box.by_id(proposal.message_id), box, folders)
        if refusals or not asks:
            continue
        message = box.by_id(proposal.message_id)
        found.append(
            {
                "message_id": proposal.message_id,
                "from": message.sender,
                "subject": message.subject,
                "action": "send a drafted reply",
                "needs_human": list(asks),
                "cites": list(proposal.cites),
                "draft": proposal.body,
            }
        )
    return found


def flagged(box, rows, folders=None):
    """Pane 2. Everything the system refused to act on.

    Three different failures share this pane because they share an outcome: the
    system declined to act and the message is still sitting there. What separates
    them is whose fault it is, so each row says what was attempted and what
    happened instead.

    The third kind was missing at first, and the gap only appeared once a standing
    instruction had been recorded. A draft the gate refuses -- for naming a time
    the owner never takes, say -- is not pending, because nobody is being asked
    about it. It was in neither pane, which made a refused send the one outcome the
    dashboard could not show.
    """
    found = []
    if folders is not None:
        for proposal in gate.proposals_from_decisions(rows, box):
            refusals, _ = gate.screen(proposal, box.by_id(proposal.message_id), box, folders)
            if not refusals:
                continue
            message = box.by_id(proposal.message_id)
            if rules.classify(message).hostile:
                continue  # reported below, with what it attempted
            found.append(
                {
                    "message_id": proposal.message_id,
                    "from": message.sender,
                    "subject": message.subject,
                    "attempted": "send the drafted reply",
                    "instead": "; ".join(refusals),
                    "kind": "refused by the gate",
                }
            )
    for threat in hostile.found(box):
        found.append(
            {
                "message_id": threat.message_id,
                "from": threat.sender,
                "subject": threat.subject,
                "attempted": threat.attempted,
                "instead": "flagged and left in place; nothing sent, moved or deleted",
                "kind": "hostile",
            }
        )
    for row in rows:
        outcome = row.get("draft_outcome")
        if outcome not in ("not_known", "rejected"):
            continue
        message = box.by_id(row.get("message_id"))
        if message is None:
            continue
        found.append(
            {
                "message_id": row["message_id"],
                "from": message.sender,
                "subject": message.subject,
                "attempted": "answer it from the inbox",
                "instead": row.get("draft_reason") or "no usable draft",
                "kind": "ungrounded" if outcome == "not_known" else "rejected",
            }
        )
    return found


def calendar(box):
    """Pane 3. What the inbox commits the owner to, with its sources and clashes."""
    found = commitments.extract(box)
    dated, undated = commitments.calendar(found)
    problems = commitments.check_citations(found, box)
    clashes = commitments.conflicts(found)

    def row(commitment):
        return {
            "what": commitment.what,
            "when": commitment.when.isoformat() if commitment.when else None,
            "when_shown": commitment.when_shown(),
            "cites": list(commitment.cites),
            "from_more_than_one": commitment.multi_source,
            "resolved_by": commitment.resolved_by,
        }

    return {
        "dated": [row(c) for c in dated],
        "undated": [row(c) for c in undated],
        "conflicts": [
            {
                "certain": clash.certain,
                "why": clash.why,
                "between": [
                    {"what": clash.first.what, "cites": list(clash.first.cites)},
                    {"what": clash.second.what, "cites": list(clash.second.cites)},
                ],
            }
            for clash in clashes
        ],
        "citation_problems": problems,
        "derived_from_more_than_one": [row(c) for c in found if c.multi_source],
    }


def build(box=None, rows=None, folders=None):
    """The whole page, as data. Nothing here renders anything."""
    box = box if box is not None else mailstore.load()
    if rows is None:
        path = config.STATE_PATH / "decisions.json"
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    folders = folders if folders is not None else actions.load_applied()
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inbox": config.INBOX_PATH.name,
        "messages": len(box.messages),
        "panes": {
            "pending": pending(box, rows, folders),
            "flagged": flagged(box, rows, folders),
            "commitments": calendar(box),
        },
    }


# --- rendering --------------------------------------------------------------


def as_text(page):
    """The terminal view. The same three panes, in the same order."""
    out = []
    panes = page["panes"]
    out.append(f"inboxHero -- {page['inbox']}, {page['messages']} messages, built {page['generated']}")

    out.append(f"\n{'=' * 78}\n1. PENDING ACTIONS -- waiting on a person ({len(panes['pending'])})\n{'=' * 78}")
    for row in panes["pending"]:
        out.append(f"\n  {row['message_id']}  {row['from']}")
        out.append(f"      {row['subject']}")
        out.append(f"      wants to: {row['action']}" + (f"  citing {', '.join(row['cites'])}" if row["cites"] else ""))
        for reason in row["needs_human"]:
            out.append(f"      needs a person because: {reason}")
    if not panes["pending"]:
        out.append("\n  (nothing is waiting)")

    out.append(f"\n{'=' * 78}\n2. FLAGGED -- refused, and left in place ({len(panes['flagged'])})\n{'=' * 78}")
    for row in panes["flagged"]:
        out.append(f"\n  {row['message_id']}  [{row['kind']}]  {row['from']}")
        out.append(f"      attempted: {row['attempted']}")
        out.append(f"      instead:   {row['instead']}")

    pane = panes["commitments"]
    out.append(f"\n{'=' * 78}\n3. COMMITMENTS -- {len(pane['dated'])} dated, {len(pane['undated'])} without a date\n{'=' * 78}")
    for row in pane["dated"]:
        mark = " **" if row["from_more_than_one"] else "   "
        out.append(f"\n{mark}{row['when_shown']:18}  {row['what']}")
        out.append(f"      from: {', '.join(row['cites'])}")
        if row["resolved_by"]:
            out.append(f"      resolved: {row['resolved_by']}")
    if pane["undated"]:
        out.append("\n  -- no date could be resolved --")
        for row in pane["undated"]:
            out.append(f"\n   {row['when_shown']:18}  {row['what']}")
            out.append(f"      from: {', '.join(row['cites'])}")

    out.append(f"\n  ** derived from more than one message ({len(pane['derived_from_more_than_one'])})")
    out.append(f"\n  conflicts ({len(pane['conflicts'])}):")
    for clash in pane["conflicts"]:
        mark = "CLASH" if clash["certain"] else "maybe"
        out.append(f"    [{mark}] {clash['why']}")
        for side in clash["between"]:
            out.append(f"        {side['what']}  [{', '.join(side['cites'])}]")
    if not pane["conflicts"]:
        out.append("    (none)")
    problems = pane["citation_problems"]
    out.append(f"\n  citations checked against the mail store: {'all good' if not problems else problems}")
    return "\n".join(out)


def as_html(page):
    """A static page, built from the JSON above and holding nothing it does not."""
    panes = page["panes"]
    e = html.escape

    def pending_rows():
        if not panes["pending"]:
            return "<tr><td colspan='3' class='none'>nothing is waiting</td></tr>"
        return "".join(
            f"<tr><td class='id'>{e(r['message_id'])}<br><span class='who'>{e(r['from'])}</span></td>"
            f"<td>{e(r['action'])}<div class='sub'>{e(r['subject'])}</div>"
            f"<details><summary>draft</summary><pre>{e(r['draft'])}</pre></details></td>"
            f"<td><ul>{''.join(f'<li>{e(x)}</li>' for x in r['needs_human'])}</ul></td></tr>"
            for r in panes["pending"]
        )

    def flagged_rows():
        return "".join(
            f"<tr><td class='id'>{e(r['message_id'])}<br><span class='tag {e(r['kind'])}'>{e(r['kind'])}</span></td>"
            f"<td>{e(r['attempted'])}<div class='sub'>{e(r['from'])}</div></td>"
            f"<td>{e(r['instead'])}</td></tr>"
            for r in panes["flagged"]
        )

    pane = panes["commitments"]

    def commitment_rows(rows):
        return "".join(
            f"<tr class=\"{'multi' if r['from_more_than_one'] else ''}\">"
            f"<td class='when'>{e(r['when_shown'])}</td>"
            f"<td>{e(r['what'])}"
            + (f"<div class='sub'>resolved: {e(r['resolved_by'])}</div>" if r["resolved_by"] else "")
            + "</td>"
            f"<td class='cites'>{' '.join(f'<code>{e(c)}</code>' for c in r['cites'])}</td></tr>"
            for r in rows
        )

    conflict_blocks = "".join(
        f"<div class=\"clash {'certain' if c['certain'] else 'maybe'}\">"
        f"<strong>{'CONFLICT' if c['certain'] else 'possible conflict'}</strong> — {e(c['why'])}"
        + "".join(
            f"<div class='side'>{e(s['what'])} <span class='cites'>"
            + " ".join(f"<code>{e(x)}</code>" for x in s["cites"])
            + "</span></div>"
            for s in c["between"]
        )
        + "</div>"
        for c in pane["conflicts"]
    ) or "<p class='none'>no two commitments collide</p>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>inboxHero</title>
<style>
  :root {{ --bg:#fbfbfa; --fg:#23211d; --muted:#6b6862; --line:#e3e0d8; --warn:#8a5a00; --bad:#8a1c1c; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#171614; --fg:#eceae4; --muted:#a09c93; --line:#2f2d28; --warn:#e0a83c; --bad:#e06a6a; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:24px 16px 64px; background:var(--bg); color:var(--fg);
         font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
  main {{ max-width:1000px; margin:0 auto; }}
  h1 {{ font-size:1.3rem; margin:0 0 4px; }}
  .meta {{ color:var(--muted); font-size:.85rem; margin-bottom:28px; }}
  h2 {{ font-size:1rem; text-transform:uppercase; letter-spacing:.08em; margin:34px 0 10px;
        padding-bottom:6px; border-bottom:2px solid var(--line); }}
  h2 span {{ color:var(--muted); font-weight:400; text-transform:none; letter-spacing:0; }}
  table {{ width:100%; border-collapse:collapse; }}
  td {{ padding:10px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
  td.id {{ white-space:nowrap; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85rem; }}
  td.when {{ white-space:nowrap; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85rem; }}
  .who, .sub {{ color:var(--muted); font-size:.82rem; font-weight:400; }}
  ul {{ margin:0; padding-left:18px; }}
  li {{ margin:2px 0; }}
  code {{ font-size:.8rem; background:var(--line); padding:1px 5px; border-radius:3px; }}
  tr.multi td {{ background:color-mix(in srgb, var(--warn) 9%, transparent); }}
  .tag {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
  .tag.hostile {{ color:var(--bad); }}
  .clash {{ border-left:3px solid var(--warn); padding:8px 12px; margin:10px 0;
            background:color-mix(in srgb, var(--warn) 8%, transparent); }}
  .clash.certain {{ border-color:var(--bad); background:color-mix(in srgb, var(--bad) 8%, transparent); }}
  .side {{ color:var(--muted); font-size:.88rem; margin-top:4px; }}
  .none {{ color:var(--muted); font-style:italic; }}
  pre {{ white-space:pre-wrap; font-size:.85rem; background:var(--line); padding:8px; border-radius:4px; }}
  details summary {{ cursor:pointer; color:var(--muted); font-size:.82rem; }}
  .key {{ color:var(--muted); font-size:.82rem; margin-top:10px; }}
</style></head><body><main>
<h1>inboxHero</h1>
<div class="meta">{e(page['inbox'])} — {page['messages']} messages — built {e(page['generated'])}</div>

<h2>1. Pending actions <span>— {len(panes['pending'])} waiting on a person</span></h2>
<table><tbody>{pending_rows()}</tbody></table>

<h2>2. Flagged <span>— {len(panes['flagged'])} refused and left in place</span></h2>
<table><tbody>{flagged_rows()}</tbody></table>

<h2>3. Commitments <span>— {len(pane['dated'])} dated, {len(pane['undated'])} unresolved</span></h2>
{conflict_blocks}
<table><tbody>{commitment_rows(pane['dated'])}</tbody></table>
{"<p class='key'>No date could be resolved for these, and none was invented.</p><table><tbody>" + commitment_rows(pane['undated']) + "</tbody></table>" if pane['undated'] else ""}
<p class="key">Shaded rows are derived from more than one message
({len(pane['derived_from_more_than_one'])} of {len(pane['dated']) + len(pane['undated'])}).
Every id was checked against the mail store: {"all good" if not pane['citation_problems'] else e(str(pane['citation_problems']))}.</p>
</main></body></html>
"""


def write(page=None):
    """Both files. The JSON first, because the page is rendered from it."""
    page = page if page is not None else build()
    config.STATE_PATH.mkdir(parents=True, exist_ok=True)
    data_path = config.STATE_PATH / "dashboard.json"
    data_path.write_text(json.dumps(page, indent=2) + "\n", encoding="utf-8")
    html_path = config.DASHBOARD_PATH
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(as_html(page), encoding="utf-8")
    return data_path, html_path


if __name__ == "__main__":
    page = build()
    print(as_text(page))
    data_path, html_path = write(page)
    print(f"\n  written to {data_path}")
    print(f"             {html_path}")
