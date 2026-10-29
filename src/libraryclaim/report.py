"""Human-readable report, built only from claim_packet.json."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, select_autoescape

from .schema import ClaimPacket

TEMPLATE = """{% macro price(x) -%}
{% if x.amount is none %}—{% else %}{{ money(x.amount) }}{% if x.converted %} (converted){% endif %}<br><a class="muted" href="{{ x.url }}">source</a>{% endif %}
{%- endmacro %}{% macro fprice(x, cur) -%}
{% if x.amount is none %}—{% else %}{{ cur }} {{ "{:,.2f}".format(x.amount) }}{% if x.converted %} (converted){% endif %}<br><a class="muted" href="{{ x.url }}">source</a>{% endif %}
{%- endmacro %}<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Contents claim {{ p.sweep.id }}</title>
<style>
body{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#1b1b1b;background:#fff}
h1{font-size:20px;margin:0 0 4px} h2{font-size:16px;margin:28px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;width:100%} th,td{border-bottom:1px solid #eee;padding:4px 6px;text-align:left;vertical-align:top}
th{background:#f6f6f6;font-weight:600} td.n{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:#777} .tag{font-size:12px;padding:1px 6px;border-radius:8px;background:#eee}
.needs_appraisal{background:#fde7c8} .unidentified{background:#eee} .identified{background:#d7f0dc}
img.t{height:56px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px}
.card{border:1px solid #eee;border-radius:6px;padding:8px} .big{font-size:18px;font-weight:600}
</style></head><body>
<h1>Contents claim: home library</h1>
<div class="muted">Sweep {{ p.sweep.id }} · {{ p.sweep.captured_at }} · {{ p.sweep.device }} · {{ p.sweep.country }} / {{ p.sweep.currency }}
 · capture: {{ p.sweep.capture_path or "n/a" }}</div>

<h2>Totals</h2>
<div class="grid">
<div class="card"><div class="muted">Books</div><div class="big">{{ t.book_count }}</div>{{ t.books_identified }} identified, {{ t.books_unidentified }} unidentified</div>
<div class="card"><div class="muted">Books, replacement</div><div class="big">{{ money(t.books_replacement_cost) }}</div>used value {{ money(t.books_used_value) }}</div>
<div class="card"><div class="muted">Other contents, replacement</div><div class="big">{{ money(t.items_replacement_cost_low) }} – {{ money(t.items_replacement_cost_high) }}</div></div>
<div class="card"><div class="muted">Excluded from totals</div><div class="big">{{ t.excluded_from_totals }}</div>awaiting appraisal or no price found</div>
<div class="card"><div class="muted">Shelf run</div><div class="big">{{ num(t.shelf_run_m, " m") }}</div></div>
</div>

<h2>Room</h2>
<table><tr><th>Length</th><th>Width</th><th>Height</th><th>Floor</th><th>Walls</th><th>Shelved wall</th><th>Scale</th><th>Confidence</th></tr>
<tr><td>{{ num(r.length_m, " m") }}</td><td>{{ num(r.width_m, " m") }}</td><td>{{ num(r.height_m, " m") }}</td>
<td>{{ num(r.floor_area_m2, " m²") }}<br><span class="muted">{{ num(r.floor_area_ft2, " ft²") }}</span></td>
<td>{{ num(r.wall_area_m2, " m²") }}<br><span class="muted">{{ num(r.wall_area_ft2, " ft²") }}</span></td>
<td>{{ num(r.shelved_wall_area_m2, " m²") }}</td><td>{{ r.scale_method or "—" }}</td><td>{{ r.confidence }}</td></tr></table>
{% if r.shape_note %}<p class="muted">{{ r.shape_note }}</p>{% endif %}
{% if r.scale_method %}<p class="muted">Measured by: {{ r.scale_method }}. Room notes are in the review queue under "room".</p>{% endif %}

<h2>Review queue ({{ p.review_queue|length }})</h2>
<table><tr><th>Line</th><th>Reason</th></tr>
{% for q in p.review_queue %}<tr><td>{{ q.ref_id }}</td><td>{{ q.reason }}</td></tr>{% endfor %}</table>

<h2>Books ({{ p.books|length }})</h2>
<table><tr><th></th><th>Line</th><th>Status</th><th>Title / author</th><th>H × T cm</th><th>Conf.</th><th>Replacement</th><th>Used</th></tr>
{% for b in p.books %}<tr>
<td>{% if b.frame_ref %}<a href="{{ b.frame_ref }}"><img class="t" src="out/crops/{{ b.id }}.jpg" alt=""></a>{% endif %}</td>
<td>{{ b.id }}<br><span class="muted">{{ b.shelf }} #{{ b.position }}</span></td>
<td><span class="tag {{ b.status }}">{{ b.status }}</span></td>
<td>{{ b.title or "—" }}<br><span class="muted">{{ b.author }} {{ b.edition }} {{ b.isbn }}</span></td>
<td class="n">{{ num(b.spine_height_cm) }} × {{ num(b.spine_thickness_cm) }}</td>
<td class="n">{{ "%.2f"|format(b.id_confidence) }}</td>
<td class="n">{{ price(b.replacement_cost) }}</td><td class="n">{{ price(b.used_value) }}</td></tr>{% endfor %}</table>

{% if p.second_country %}
<h2>Same books priced for {{ p.second_country.country }} ({{ p.second_country.currency }})</h2>
<p class="muted">The pricing code with a different country setting, to show locale is not hardcoded. Not included in the totals.</p>
<table><tr><th>Line</th><th>Title</th><th>Replacement here</th><th>Replacement in {{ p.second_country.country }}</th><th>Used in {{ p.second_country.country }}</th></tr>
{% for l in p.second_country.lines %}{% set home = p.books | selectattr("id", "equalto", l.book_id) | first %}<tr><td>{{ l.book_id }}</td><td>{{ l.title }}</td>
<td class="n">{{ price(home.replacement_cost) if home else "—" }}</td>
<td class="n">{{ fprice(l.replacement_cost, p.second_country.currency) }}</td><td class="n">{{ fprice(l.used_value, p.second_country.currency) }}</td></tr>{% endfor %}</table>
{% endif %}

<h2>Other contents ({{ p.items|length }})</h2>
<table><tr><th>Line</th><th>Category</th><th>Description</th><th>Approx. size</th><th>Status</th><th>Replacement</th></tr>
{% for i in p.items %}<tr><td>{% if i.frame_ref %}<a href="{{ i.frame_ref }}">{{ i.id }}</a>{% else %}{{ i.id }}{% endif %}</td>
<td>{{ i.category }}</td><td>{{ i.description }}<br><span class="muted">{{ i.brand_model }}</span></td>
<td class="n">{% if i.dimensions_cm.w %}{{ "%g"|format(i.dimensions_cm.w) }} × {{ "%g"|format(i.dimensions_cm.h) }} cm<br><span class="muted">front face</span>{% else %}—{% endif %}</td>
<td><span class="tag {{ i.status }}">{{ i.status }}</span></td>
<td class="n">{% if i.replacement_cost.low is not none %}{{ money(i.replacement_cost.low) }}{% if i.replacement_cost.high != i.replacement_cost.low %} – {{ money(i.replacement_cost.high) }}{% endif %}
<br><a class="muted" href="{{ i.replacement_cost.url }}">{{ i.replacement_cost.source }}</a>{% else %}—{% endif %}</td></tr>{% endfor %}</table>
<p class="muted">Every price links to the listing search it came from, with its retrieval date in claim_packet.json. Asking prices of active listings; totals are computed from the lines.</p>
</body></html>"""


def render(packet: ClaimPacket) -> str:
    cur = packet.sweep.currency or ""

    def money(v):
        return "—" if v is None else f"{cur} {v:,.2f}"

    def num(v, unit=""):
        return "—" if v is None else f"{v:g}{unit}"

    env = Environment(autoescape=select_autoescape(default=True, default_for_string=True))
    env.globals.update(money=money, num=num)
    return env.from_string(TEMPLATE).render(p=packet, t=packet.totals, r=packet.room)


def write(packet: ClaimPacket, root: Path) -> Path:
    path = root / "report.html"
    path.write_text(render(packet))
    return path
