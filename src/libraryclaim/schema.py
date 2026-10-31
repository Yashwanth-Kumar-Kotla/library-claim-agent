"""claim_packet.json contract. Field names follow the brief exactly; extra fields are additive."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Sweep(BaseModel):
    id: str = ""
    captured_at: str = ""
    device: str = ""
    duration_s: float = 0
    country: str = ""
    currency: str = ""
    capture_path: Literal["browser", "ios_lidar", ""] = ""


class Room(BaseModel):
    length_m: Optional[float] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    floor_area_m2: Optional[float] = None
    wall_area_m2: Optional[float] = None
    shelved_wall_area_m2: Optional[float] = None
    floor_area_ft2: Optional[float] = None
    wall_area_ft2: Optional[float] = None
    shape_note: str = ""
    scale_method: str = ""
    confidence: float = 0


class BookPrice(BaseModel):
    amount: Optional[float] = None
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    converted: bool = False
    condition_assumed: str = ""


class Book(BaseModel):
    id: str
    shelf: str = ""
    position: int = 0
    frame_ref: str = ""
    status: Literal["identified", "unidentified", "needs_appraisal"] = "unidentified"
    title: str = ""
    author: str = ""
    publisher: str = ""  # as printed on the spine; empty when not legible
    edition: str = ""
    isbn: str = ""
    spine_height_cm: Optional[float] = None
    spine_thickness_cm: Optional[float] = None
    id_confidence: float = 0
    replacement_cost: BookPrice = Field(default_factory=BookPrice)
    used_value: BookPrice = Field(default_factory=BookPrice)


class Dimensions(BaseModel):
    w: Optional[float] = None
    h: Optional[float] = None
    d: Optional[float] = None


class ItemPrice(BaseModel):
    low: Optional[float] = None
    high: Optional[float] = None
    source: str = ""
    url: str = ""
    retrieved_at: str = ""


class Item(BaseModel):
    id: str
    category: str = ""
    description: str = ""
    brand_model: str = ""
    frame_ref: str = ""
    dimensions_cm: Dimensions = Field(default_factory=Dimensions)
    status: Literal["priced", "range", "needs_appraisal"] = "range"
    replacement_cost: ItemPrice = Field(default_factory=ItemPrice)
    confidence: float = 0


class Totals(BaseModel):
    book_count: int = 0
    books_identified: int = 0
    books_unidentified: int = 0
    shelf_run_m: Optional[float] = None
    books_replacement_cost: float = 0
    books_used_value: float = 0
    items_replacement_cost_low: float = 0
    items_replacement_cost_high: float = 0
    excluded_from_totals: int = 0


class ReviewEntry(BaseModel):
    ref_id: str
    reason: str


class SecondCountryLine(BaseModel):
    book_id: str
    title: str = ""
    isbn: str = ""
    replacement_cost: BookPrice = Field(default_factory=BookPrice)
    used_value: BookPrice = Field(default_factory=BookPrice)
    note: str = ""


class SecondCountry(BaseModel):
    """The same books priced for another country, to show locale is a setting. Not in the totals."""
    country: str
    currency: str
    lines: list[SecondCountryLine] = Field(default_factory=list)


class ClaimPacket(BaseModel):
    sweep: Sweep = Field(default_factory=Sweep)
    room: Room = Field(default_factory=Room)
    books: list[Book] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    review_queue: list[ReviewEntry] = Field(default_factory=list)
    second_country: SecondCountry | None = None


def compute_totals(packet: ClaimPacket, shelf_run_m: Optional[float] = None) -> Totals:
    """Totals are summed here from the lines, never written by a model.

    A line counts toward a money total only if it has an amount and is not awaiting appraisal.
    Every other line is counted in excluded_from_totals.
    """
    books_repl = books_used = items_low = items_high = 0.0
    excluded = 0
    for b in packet.books:
        if b.status == "needs_appraisal" or b.replacement_cost.amount is None:
            excluded += 1
            continue
        books_repl += b.replacement_cost.amount
        if b.used_value.amount is not None:
            books_used += b.used_value.amount
    for it in packet.items:
        if it.status == "needs_appraisal" or it.replacement_cost.low is None:
            excluded += 1
            continue
        items_low += it.replacement_cost.low
        items_high += it.replacement_cost.high if it.replacement_cost.high is not None else it.replacement_cost.low
    identified = sum(b.status == "identified" for b in packet.books)
    return Totals(
        book_count=len(packet.books),
        books_identified=identified,
        books_unidentified=sum(b.status == "unidentified" for b in packet.books),
        shelf_run_m=shelf_run_m,
        books_replacement_cost=round(books_repl, 2),
        books_used_value=round(books_used, 2),
        items_replacement_cost_low=round(items_low, 2),
        items_replacement_cost_high=round(items_high, 2),
        excluded_from_totals=excluded,
    )
