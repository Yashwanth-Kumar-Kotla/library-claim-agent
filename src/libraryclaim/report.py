"""Human-readable report, built only from claim_packet.json.

A claim dossier for an adjuster: headline figures, how the count splits, the room with a floor plan,
everything that needs a human, then the ledger of books and other contents with every price linked
to the listing search it came from.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from jinja2 import Environment, select_autoescape

from .schema import ClaimPacket

TEMPLATE = """{% macro price(x, cur) -%}
<span class="v">{% if x.amount is none %}<span class="nil">—</span>{% else %}<span class="amt">{{ money(x.amount, cur) }}</span>{% if x.converted %}<span class="conv" title="converted from another market">conv</span>{% endif %}
<a class="src" href="{{ x.url }}" target="_blank" rel="noopener">source</a>{% endif %}</span>
{%- endmacro %}<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Contents claim {{ p.sweep.id }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--paper:#f5f0e6;--card:#fbf8f2;--ink:#1d1a15;--soft:#5d564b;--faint:#8f8778;--rule:#e2d9c8;--brass:#a8792a;
--sage:#3f7a4f;--sage-bg:#e3efe3;--amber:#b0611a;--amber-bg:#f8e7d2;--grey-bg:#ece6da;--rose:#a8453b;
--serif:"Hanken Grotesk",-apple-system,"Segoe UI",sans-serif;--sans:"Hanken Grotesk",-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--paper:#14120f;--card:#1c1a16;--ink:#eee7da;--soft:#b7ae9e;--faint:#857d6f;--rule:#2e2a23;
--brass:#d4ad5b;--sage:#9fc6a3;--sage-bg:#1f2d22;--amber:#f0a453;--amber-bg:#33251a;--grey-bg:#26231d;--rose:#e88f86}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 var(--sans);
background-image:radial-gradient(90% 40% at 100% 0%,color-mix(in srgb,var(--brass) 9%,transparent),transparent 70%)}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 72px}
a{color:inherit}
.eyebrow{font:500 11.5px/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--brass);margin:0 0 14px}
h1{font:700 clamp(30px,4.6vw,46px)/1.1 var(--serif);letter-spacing:-.025em;margin:0 0 12px}
h1 em{font-style:normal;color:var(--brass)}
h2{font:700 21px/1.2 var(--serif);margin:0;letter-spacing:-.015em}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--soft);font:400 13px var(--mono)}
.meta b{color:var(--ink);font-weight:500}
section{margin-top:44px}
.head{display:flex;align-items:baseline;justify-content:space-between;gap:16px;border-bottom:1px solid var(--ink);padding-bottom:10px;margin-bottom:18px;flex-wrap:wrap}
.head .note{color:var(--faint);font-size:13px}

.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:16px;overflow:hidden;margin-top:30px}
.fig{background:var(--card);padding:20px 20px 18px;animation:rise .6s both}
.fig:nth-child(2){animation-delay:.05s}.fig:nth-child(3){animation-delay:.1s}.fig:nth-child(4){animation-delay:.15s}
.fig small{display:block;font:500 11px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.fig b{display:block;font:700 30px/1.15 var(--serif);margin:8px 0 4px;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.fig span{color:var(--soft);font-size:13px}
.fig.alert b{color:var(--amber)}
.fig b.range{font-size:21px;line-height:1.75;white-space:nowrap}

.split{margin-top:22px}
.bar{display:flex;height:12px;border-radius:6px;overflow:hidden;background:var(--grey-bg)}
.bar i{display:block}
.bar .s-id{background:var(--sage)}.bar .s-ap{background:var(--amber)}.bar .s-un{background:var(--faint);opacity:.45}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:10px;font-size:13px;color:var(--soft)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:0}

.room{display:grid;grid-template-columns:minmax(220px,340px) 1fr;gap:28px;align-items:start}
@media(max-width:720px){.room{grid-template-columns:1fr}}
.plan{background:var(--card);border:1px solid var(--rule);border-radius:16px;padding:18px}
.plan svg{width:100%;height:auto;display:block}
.plan .wall{fill:color-mix(in srgb,var(--brass) 10%,transparent);stroke:var(--ink);stroke-width:2.5}
.plan text{font:500 11px var(--mono);fill:var(--soft)}
.plan .empty{color:var(--faint);font-size:14px;padding:40px 10px;text-align:center}
dl.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px 22px;margin:0}
dl.stats div{border-top:1px solid var(--rule);padding-top:10px}
dt{font:500 11px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
dd{margin:4px 0 0;font:600 21px/1.2 var(--serif);font-variant-numeric:tabular-nums}
dd small{display:block;font:400 12.5px var(--sans);color:var(--soft)}
.method{margin-top:16px;color:var(--soft);font-size:13.5px}
.meter{display:inline-block;width:70px;height:6px;border-radius:3px;background:var(--grey-bg);vertical-align:middle;margin-left:6px;overflow:hidden}
.meter i{display:block;height:100%;background:var(--brass)}

.queue{display:grid;grid-template-columns:minmax(0,1fr);gap:8px}
details{background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:0 16px}
summary{list-style:none;cursor:pointer;display:flex;gap:12px;align-items:center;padding:12px 0}
summary::-webkit-details-marker{display:none}
summary .ref{font:500 13px var(--mono);min-width:70px}
summary .first,.one .first{flex:1;min-width:0;color:var(--soft);font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.one{display:flex;gap:12px;align-items:baseline;background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:12px 16px}
.one .ref{font:500 13px var(--mono);min-width:70px}.one .first{white-space:normal}
summary .n{font:500 12px var(--mono);color:var(--faint)}
summary::after{content:"+";font:400 18px var(--mono);color:var(--faint)}
details[open] summary::after{content:"−"}
details ul{margin:0 0 14px;padding-left:82px;color:var(--soft);font-size:14px}
@media(max-width:560px){details ul{padding-left:18px}summary .ref,.one .ref{min-width:0}}
html,body{overflow-x:hidden}
details li{margin:4px 0}

.tools{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
.f{border:1px solid var(--rule);background:var(--card);color:var(--soft);border-radius:999px;padding:6px 13px;font:500 13px var(--sans);cursor:pointer}
.f.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.f b{font-family:var(--mono);font-weight:500;margin-left:4px;opacity:.75}
.search{margin-left:auto;border:1px solid var(--rule);background:var(--card);color:var(--ink);border-radius:10px;padding:8px 12px;font:14px var(--sans);min-width:200px}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:14px;background:var(--card)}
table{border-collapse:collapse;width:100%;min-width:760px}
th{font:500 11px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--faint);text-align:left;padding:12px 14px;border-bottom:1px solid var(--rule);position:sticky;top:0;background:var(--card)}
td{padding:11px 14px;border-bottom:1px solid var(--rule);vertical-align:top}
tr:last-child td{border-bottom:0}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.thumb{width:34px;height:52px;object-fit:cover;border-radius:4px;background:var(--grey-bg);display:block}
.lid{font:400 12.5px var(--mono);color:var(--faint);white-space:nowrap}
.ttl{font:600 15px/1.3 var(--serif);letter-spacing:-.005em}
.sub{color:var(--soft);font-size:13px}
.tag{display:inline-block;font:500 10.5px var(--mono);letter-spacing:.06em;text-transform:uppercase;padding:3px 8px;border-radius:6px;white-space:nowrap}
.t-identified,.t-priced{color:var(--sage);background:var(--sage-bg)}
.t-needs_appraisal{color:var(--amber);background:var(--amber-bg)}
.t-unidentified,.t-range{color:var(--soft);background:var(--grey-bg)}
.amt{font-variant-numeric:tabular-nums;white-space:nowrap}
.src{display:block;font:400 11.5px var(--mono);color:var(--faint);text-decoration-thickness:1px}
.conv{font:500 10px var(--mono);color:var(--amber);margin-left:6px;text-transform:uppercase}
.nil{color:var(--faint)}
.conf{display:inline-block;width:46px;height:5px;border-radius:3px;background:var(--grey-bg);overflow:hidden;vertical-align:middle;margin-right:6px}
.conf i{display:block;height:100%;background:var(--ink);opacity:.7}
tr.hide{display:none!important}
@media(max-width:640px){
  table.cards{min-width:0}.scroll:has(table.cards){overflow:visible}
  table.cards tr:first-child{display:none}
  table.cards tr{display:grid;grid-template-columns:auto 1fr;column-gap:12px;row-gap:4px;padding:14px;border-bottom:1px solid var(--rule)}
  table.cards tr:last-child{border-bottom:0}
  table.cards td{display:block;padding:0;border:0;text-align:left}
  table.cards td{grid-column:2}
  table.cards td:first-child{grid-column:1;grid-row:1 / span 7}
  table.cards td[data-l]{display:flex;justify-content:space-between;align-items:baseline;gap:12px;font-size:14px}
  table.cards td[data-l] .v{text-align:right}
  table.cards td[data-l]::before{content:attr(data-l);font:500 10.5px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--faint);padding-top:3px}
  table.cards .src{display:inline;margin-left:8px}
}
footer{margin-top:56px;padding-top:16px;border-top:1px solid var(--rule);color:var(--faint);font-size:13px;display:grid;gap:6px}
@keyframes rise{from{transform:translateY(8px)}to{transform:none}}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
@media print{body{background:#fff}.tools,.src{display:none}.scroll{overflow:visible;border:0}section{break-inside:avoid-page}details{break-inside:avoid}}
</style></head><body><div class="wrap">

<header>
  <p class="eyebrow">Contents claim · home library</p>
  <h1>{{ t.book_count }} books, {{ p.items|length }} other items{% if r.floor_area_m2 %}, <em>{{ num(r.floor_area_m2) }} m²</em>{% endif %}</h1>
  <div class="meta">
    <span>Sweep <b>{{ p.sweep.id or "—" }}</b></span><span>{{ p.sweep.captured_at or "" }}</span>
    <span>{{ p.sweep.device or "" }}</span><span>Values in <b>{{ p.sweep.currency or "—" }}</b> · {{ p.sweep.country }}</span>
    {% if p.sweep.duration_s %}<span>{{ "%.0f"|format(p.sweep.duration_s) }} s sweep</span>{% endif %}
  </div>
</header>

<div class="figs">
  <div class="fig"><small>Books · replacement</small><b>{{ money(t.books_replacement_cost) }}</b><span>used value {{ money(t.books_used_value) }}</span></div>
  <div class="fig"><small>Other contents</small><b class="{% if t.items_replacement_cost_high != t.items_replacement_cost_low %}range{% endif %}">{{ money(t.items_replacement_cost_low) }}{% if t.items_replacement_cost_high != t.items_replacement_cost_low %} – {{ "{:,.2f}".format(t.items_replacement_cost_high) }}{% endif %}</b><span>replacement, low to high</span></div>
  <div class="fig {% if t.excluded_from_totals %}alert{% endif %}"><small>Excluded from totals</small><b>{{ t.excluded_from_totals }}</b><span>awaiting appraisal or no price found</span></div>
  <div class="fig {% if groups %}alert{% endif %}"><small>For the adjuster</small><b>{{ groups|length }}</b><span>lines with review notes</span></div>
</div>

{% if t.book_count %}
<div class="split">
  <div class="bar">
    <i class="s-id" style="width:{{ pct(n_id) }}%"></i><i class="s-ap" style="width:{{ pct(n_ap) }}%"></i><i class="s-un" style="width:{{ pct(n_un) }}%"></i>
  </div>
  <div class="legend">
    <span><i style="background:var(--sage)"></i>{{ n_id }} identified</span>
    <span><i style="background:var(--amber)"></i>{{ n_ap }} need appraisal</span>
    <span><i style="background:var(--faint);opacity:.45"></i>{{ n_un }} unidentified</span>
    <span>shelf run {{ num(t.shelf_run_m, " m") }}</span>
  </div>
</div>
{% endif %}

<section>
  <div class="head"><h2>Room</h2><span class="note">{{ r.shape_note or "" }}</span></div>
  <div class="room">
    <div class="plan">{% if plan %}
      <svg viewBox="0 0 {{ plan.vw }} {{ plan.vh }}" role="img" aria-label="Floor plan {{ num(r.length_m) }} by {{ num(r.width_m) }} metres">
        <rect class="wall" x="{{ plan.x }}" y="{{ plan.y }}" width="{{ plan.w }}" height="{{ plan.h }}" rx="2"/>
        <text x="{{ plan.x + plan.w / 2 }}" y="{{ plan.y - 8 }}" text-anchor="middle">{{ num(r.length_m) }} m</text>
        <text x="{{ plan.x + plan.w + 8 }}" y="{{ plan.y + plan.h / 2 }}" dominant-baseline="middle">{{ num(r.width_m) }} m</text>
        <text x="{{ plan.x + plan.w / 2 }}" y="{{ plan.y + plan.h / 2 }}" text-anchor="middle" dominant-baseline="middle">{{ num(r.floor_area_m2) }} m²</text>
      </svg>{% else %}<div class="empty">Room size was not measured in this sweep.</div>{% endif %}
    </div>
    <div>
      <dl class="stats">
        <div><dt>Floor</dt><dd>{{ num(r.floor_area_m2, " m²") }}<small>{{ num(r.floor_area_ft2, " ft²") }}</small></dd></div>
        <div><dt>Walls</dt><dd>{{ num(r.wall_area_m2, " m²") }}<small>{{ num(r.wall_area_ft2, " ft²") }}</small></dd></div>
        <div><dt>Shelved wall</dt><dd>{{ num(r.shelved_wall_area_m2, " m²") }}<small>covered by shelving</small></dd></div>
        <div><dt>Length × width</dt><dd>{{ num(r.length_m) }} × {{ num(r.width_m) }}<small>metres</small></dd></div>
        <div><dt>Ceiling</dt><dd>{{ num(r.height_m, " m") }}<small>floor to ceiling</small></dd></div>
      </dl>
      {% if r.scale_method %}<p class="method">Measured by {{ r.scale_method }}. Confidence <span class="meter"><i style="width:{{ (r.confidence * 100)|round|int }}%"></i></span></p>{% endif %}
    </div>
  </div>
</section>

<section>
  <div class="head"><h2>For the adjuster</h2><span class="note">every line the system is not certain of, and why</span></div>
  {% if groups %}<div class="queue">{% for ref, reasons in groups.items() %}{% if reasons|length == 1 %}
    <div class="one"><span class="ref">{{ ref }}</span><span class="first">{{ reasons[0] }}</span></div>{% else %}
    <details><summary><span class="ref">{{ ref }}</span><span class="first">{{ reasons[0] }}</span><span class="n">{{ reasons|length }}</span></summary>
    <ul>{% for x in reasons %}<li>{{ x }}</li>{% endfor %}</ul></details>{% endif %}{% endfor %}</div>
  {% else %}<p class="sub">Nothing needs review.</p>{% endif %}
</section>

<section>
  <div class="head"><h2>Books</h2><span class="note">prices are medians of live listings; each links to its search</span></div>
  <div class="tools" data-table="books">
    <button class="f on" data-f="all">All<b>{{ p.books|length }}</b></button>
    <button class="f" data-f="identified">Identified<b>{{ n_id }}</b></button>
    <button class="f" data-f="needs_appraisal">Appraisal<b>{{ n_ap }}</b></button>
    <button class="f" data-f="unidentified">Unidentified<b>{{ n_un }}</b></button>
    <input class="search" type="search" placeholder="Search title, author, line" aria-label="Search books">
  </div>
  <div class="scroll"><table id="books" class="cards">
    <tr><th></th><th>Line</th><th>Book</th><th>Status</th><th class="n">H × T cm</th><th>Match</th><th class="n">Replacement</th><th class="n">Used</th></tr>
    {% for b in p.books %}<tr data-s="{{ b.status }}" data-q="{{ (b.id ~ ' ' ~ b.title ~ ' ' ~ b.author)|lower }}">
      <td>{% if b.frame_ref %}<a href="{{ b.frame_ref }}" target="_blank" rel="noopener"><img class="thumb" loading="lazy" src="out/crops/{{ b.id }}.jpg" alt=""></a>{% endif %}</td>
      <td class="lid">{{ b.id }}</td>
      <td>{% if b.title %}<div class="ttl">{{ b.title }}</div><div class="sub">{{ b.author }}{% if b.edition %} · {{ b.edition }}{% endif %}{% if b.isbn %} · ISBN {{ b.isbn }}{% endif %}</div>{% else %}<div class="sub">Spine not identified</div>{% endif %}</td>
      <td><span class="tag t-{{ b.status }}">{{ b.status|replace("_", " ") }}</span></td>
      <td class="n" data-l="Height × thickness"><span class="v">{{ num(b.spine_height_cm) }} × {{ num(b.spine_thickness_cm) }} cm</span></td>
      <td data-l="Match"><span class="v">{% if b.id_confidence %}<span class="conf"><i style="width:{{ (b.id_confidence * 100)|round|int }}%"></i></span><span class="lid">{{ "%.2f"|format(b.id_confidence) }}</span>{% else %}<span class="nil">—</span>{% endif %}</span></td>
      <td class="n" data-l="Replacement">{{ price(b.replacement_cost, cur) }}</td><td class="n" data-l="Used">{{ price(b.used_value, cur) }}</td>
    </tr>{% endfor %}
  </table></div>
</section>

{% if p.second_country %}
<section>
  <div class="head"><h2>Same books in {{ p.second_country.country }}</h2><span class="note">the same pricing code with another country setting · not in the totals</span></div>
  <div class="scroll"><table>
    <tr><th>Line</th><th>Book</th><th class="n">Replacement · {{ cur }}</th><th class="n">Replacement · {{ p.second_country.currency }}</th><th class="n">Used · {{ p.second_country.currency }}</th></tr>
    {% for l in p.second_country.lines %}{% set home = p.books | selectattr("id", "equalto", l.book_id) | first %}<tr>
      <td class="lid">{{ l.book_id }}</td><td><div class="ttl">{{ l.title }}</div>{% if l.note %}<div class="sub">{{ l.note }}</div>{% endif %}</td>
      <td class="n">{{ price(home.replacement_cost, cur) if home else "—" }}</td>
      <td class="n">{{ price(l.replacement_cost, p.second_country.currency) }}</td><td class="n">{{ price(l.used_value, p.second_country.currency) }}</td>
    </tr>{% endfor %}
  </table></div>
</section>
{% endif %}

<section>
  <div class="head"><h2>Other contents</h2><span class="note">sizes are the front face, from depth; art waits for appraisal unless it is a print</span></div>
  {% if p.items %}<div class="scroll"><table class="cards">
    <tr><th>Line</th><th>Item</th><th>Category</th><th class="n">W × H cm</th><th>Status</th><th class="n">Replacement</th></tr>
    {% for i in p.items %}<tr>
      <td class="lid">{% if i.frame_ref %}<a href="{{ i.frame_ref }}" target="_blank" rel="noopener">{{ i.id }}</a>{% else %}{{ i.id }}{% endif %}</td>
      <td><div class="ttl">{{ i.description or i.category }}</div>{% if i.brand_model %}<div class="sub">{{ i.brand_model }}</div>{% endif %}</td>
      <td class="sub" data-l="Category"><span class="v">{{ i.category }}</span></td>
      <td class="n" data-l="Front, W × H"><span class="v">{% if i.dimensions_cm.w %}{{ "%g"|format(i.dimensions_cm.w) }} × {{ "%g"|format(i.dimensions_cm.h) }}{% else %}<span class="nil">—</span>{% endif %}</span></td>
      <td><span class="tag t-{{ i.status }}">{{ i.status|replace("_", " ") }}</span></td>
      <td class="n" data-l="Replacement"><span class="v">{% if i.replacement_cost.low is not none %}<span class="amt">{{ money(i.replacement_cost.low) }}{% if i.replacement_cost.high != i.replacement_cost.low %} – {{ money(i.replacement_cost.high) }}{% endif %}</span>
        <a class="src" href="{{ i.replacement_cost.url }}" target="_blank" rel="noopener">source</a>{% else %}<span class="nil">—</span>{% endif %}</span></td>
    </tr>{% endfor %}
  </table></div>{% else %}<p class="sub">No other contents recorded.</p>{% endif %}
</section>

<footer>
  <span>Totals are computed from the lines above. A line without a price source, link and retrieval date carries no price.</span>
  <span>Each book and item links to the saved frame it was seen in; retrieval dates are in claim_packet.json.</span>
</footer>
</div>
<script>
document.querySelectorAll(".tools").forEach(function (bar) {
  var table = document.getElementById(bar.dataset.table), filter = "all", q = "";
  function apply() {
    table.querySelectorAll("tr[data-s]").forEach(function (tr) {
      var ok = (filter === "all" || tr.dataset.s === filter) && (!q || tr.dataset.q.indexOf(q) !== -1);
      tr.classList.toggle("hide", !ok);
    });
  }
  bar.querySelectorAll(".f").forEach(function (b) {
    b.onclick = function () { bar.querySelectorAll(".f").forEach(function (x) { x.classList.remove("on"); }); b.classList.add("on"); filter = b.dataset.f; apply(); };
  });
  bar.querySelector(".search").oninput = function (e) { q = e.target.value.trim().toLowerCase(); apply(); };
});
</script>
</body></html>"""


def floor_plan(length: float | None, width: float | None) -> dict | None:
    """Rectangle for the SVG floor plan, room proportions kept, fitted to a fixed box."""
    if not length or not width:
        return None
    vw, vh, pad = 300, 220, 40
    s = min((vw - 2 * pad) / length, (vh - 2 * pad) / width)
    w, h = length * s, width * s
    return {"vw": vw, "vh": vh, "w": round(w, 1), "h": round(h, 1), "x": round((vw - w) / 2 - 10, 1), "y": round((vh - h) / 2 + 6, 1)}


def render(packet: ClaimPacket) -> str:
    cur = packet.sweep.currency or ""

    def money(v, currency=None):
        return "—" if v is None else f"{currency or cur} {v:,.2f}"

    def num(v, unit=""):
        return "—" if v is None else f"{v:g}{unit}"

    counts = {s: sum(b.status == s for b in packet.books) for s in ("identified", "needs_appraisal", "unidentified")}
    total = max(1, len(packet.books))
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for entry in sorted(packet.review_queue, key=lambda e: (e.ref_id not in ("room", "sweep"), e.ref_id)):
        groups.setdefault(entry.ref_id, []).append(entry.reason)

    env = Environment(autoescape=select_autoescape(default=True, default_for_string=True))
    env.globals.update(money=money, num=num, pct=lambda n: round(100 * n / total, 2))
    return env.from_string(TEMPLATE).render(
        p=packet, t=packet.totals, r=packet.room, cur=cur, groups=groups,
        n_id=counts["identified"], n_ap=counts["needs_appraisal"], n_un=counts["unidentified"],
        plan=floor_plan(packet.room.length_m, packet.room.width_m))


def write(packet: ClaimPacket, root: Path) -> Path:
    path = root / "report.html"
    path.write_text(render(packet))
    return path
