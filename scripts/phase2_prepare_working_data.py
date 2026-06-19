#!/usr/bin/env python3
"""Prepare Phase 2 care home working data for the Streamlit dashboard.

The script is intentionally conservative:
- raw files are read-only inputs under local_data/phase2_raw
- generated outputs are written under local_data/phase2_processed
- the old working workbook is used only as a schema and metadata reference
"""

from __future__ import annotations

import argparse
import difflib
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE2_RAW_DIR = (
    ROOT
    / "local_data"
    / "phase2_raw"
    / "Stage_2_THA_LU"
    / "Stage2 Data from Tara"
    / "Phase 2 data"
)
DEFAULT_OUTPUT_DIR = ROOT / "local_data" / "phase2_processed" / "2026-06-12"
DEFAULT_OLD_WORKING_DATA = ROOT / "Data" / "New_Area_Working_Data.xlsx"
DEFAULT_OBSERVATION_WORKBOOK = (
    PHASE2_RAW_DIR
    / "escalation data"
    / "rutland leicester news2 summary data, 05 may 2026.xlsx"
)
DEFAULT_QUARTERLY_WORKBOOK = PHASE2_RAW_DIR / "Whzan LLR Quarterly March 2026.xlsx"
DEFAULT_ODS_WORKBOOK = PHASE2_RAW_DIR / "ODS codes" / "05_May_2026_HSCA_Active_Locations.ods"


OLD_SCHEMA_COLUMNS = [
    "Date/Time",
    "Care Home ID",
    "Care Home Name",
    "Type",
    "Area",
    "Phase",
    "Postal address",
    "Post Code",
    "GP",
    "GP Practice",
    "GP Postcode",
    "Type of service",
    "No of Beds",
    "Miles from THA - LE12 8FE",
    "Asset number",
    "Amount of Asset",
    "Weekly fee",
    "Provider company",
    "Clinical concern?",
    "NEWS2 score",
    "New2 Score_New",
    "H=I?",
    "O2",
    "O2_New",
    "Systolic",
    "Systolic_New",
    "Diasolic",
    "Pulse",
    "Pulse_New",
    "Temperature",
    "Temperate_New",
    "Respiration rate",
    "Respiraties_New",
    "O2 delivery",
    "O2 Delivery_New",
    "Consciousness",
    "Consciouness New",
    "Scale 2 in use?",
]

EXTRA_WORKING_COLUMNS = [
    "Source Caseload",
    "Source Sheet",
    "Source NEWS2 score",
    "NEWS2 Score Difference",
    "ODS Match Status",
    "ODS Location ID",
    "ODS Code",
    "ODS Location Name",
    "ODS Beds",
    "Data Quality Flags",
]

REQUIRED_VITAL_FIELDS = ["O2", "Systolic", "Pulse", "Temperature", "Respiration rate"]
LOCAL_AUTHORITY_PATTERN = re.compile(r"Leicester|Leicestershire|Rutland", re.IGNORECASE)
OUT_OF_SCOPE_CAREHOME_IDS = {"1082"}


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    working_data: Path
    audit_report: Path
    summary_markdown: Path


def clean_column_names(columns: pd.Index) -> list[str]:
    return [re.sub(r"\s+", " ", str(col)).strip() for col in columns]


def clean_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def clean_id(value: Any) -> str:
    text = clean_string(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def clean_postcode(value: Any) -> str:
    text = clean_string(value).upper()
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text)
    if len(compact) > 3:
        return compact[:-3] + " " + compact[-3:]
    return compact


def postcode_key(value: Any) -> str:
    return re.sub(r"\s+", "", clean_string(value).upper())


def normalize_name(value: Any) -> str:
    text = clean_string(value).lower().replace("&", " and ")
    text = re.sub(r"\b(care home|nursing home|residential home|carehome|ltd|limited)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def to_number(value: Any) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return np.nan
    return float(parsed)


def score_respiration(value: Any) -> float:
    x = to_number(value)
    if pd.isna(x) or x <= 0:
        return np.nan
    if x <= 8:
        return 3
    if x <= 11:
        return 1
    if x <= 20:
        return 0
    if x <= 24:
        return 2
    return 3


def scale2_in_use(value: Any) -> bool:
    text = clean_string(value).lower()
    return text in {"1", "true", "yes", "y", "scale 2"}


def oxygen_on_air(value: Any) -> bool:
    text = clean_string(value).lower()
    return "air" in text or "breathing" in text


def score_oxygen_saturation(o2_value: Any, o2_delivery: Any, scale2_value: Any) -> float:
    x = to_number(o2_value)
    if pd.isna(x) or x <= 0:
        return np.nan

    if not scale2_in_use(scale2_value):
        if x <= 91:
            return 3
        if x <= 93:
            return 2
        if x <= 95:
            return 1
        return 0

    if x <= 83:
        return 3
    if x <= 85:
        return 2
    if x <= 87:
        return 1
    if x <= 92:
        return 0
    if oxygen_on_air(o2_delivery):
        return 0
    if x <= 94:
        return 1
    if x <= 96:
        return 2
    return 3


def score_systolic(value: Any) -> float:
    x = to_number(value)
    if pd.isna(x) or x <= 0:
        return np.nan
    if x <= 90:
        return 3
    if x <= 100:
        return 2
    if x <= 110:
        return 1
    if x <= 219:
        return 0
    return 3


def score_pulse(value: Any) -> float:
    x = to_number(value)
    if pd.isna(x) or x <= 0:
        return np.nan
    if x <= 40:
        return 3
    if x <= 50:
        return 1
    if x <= 90:
        return 0
    if x <= 110:
        return 1
    if x <= 130:
        return 2
    return 3


def score_temperature(value: Any) -> float:
    x = to_number(value)
    if pd.isna(x) or x <= 0:
        return np.nan
    if x <= 35.0:
        return 3
    if x <= 36.0:
        return 1
    if x <= 38.0:
        return 0
    if x <= 39.0:
        return 1
    return 2


def score_o2_delivery(value: Any) -> float:
    text = clean_string(value)
    if not text:
        return np.nan
    return 0 if oxygen_on_air(text) else 2


def score_consciousness(value: Any) -> float:
    text = clean_string(value).lower()
    if not text:
        return np.nan
    return 0 if text == "alert" else 3


def load_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, dtype=object)
    df.columns = clean_column_names(df.columns)
    return df.dropna(how="all").copy()


def combine_date_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    date_part = pd.to_datetime(date_series, errors="coerce")
    time_text = time_series.map(clean_string)
    combined = pd.to_datetime(date_part.dt.strftime("%Y-%m-%d") + " " + time_text, errors="coerce")
    return combined.fillna(date_part)


def build_old_metadata(old_df: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_name: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for _, row in old_df.dropna(subset=["Care Home Name"]).iterrows():
        record = row.to_dict()
        name_key = normalize_name(row.get("Care Home Name"))
        id_key = clean_id(row.get("Care Home ID"))
        if name_key and name_key not in by_name:
            by_name[name_key] = record
        if id_key and id_key not in by_id:
            by_id[id_key] = record
    return by_name, by_id


def choose_mapping_row(group: pd.DataFrame) -> pd.Series:
    scored = group.copy()
    month = pd.to_datetime(scored.get("Month"), errors="coerce")
    scored["_mapping_score"] = 0.0
    scored["_mapping_score"] += scored.get("Carehome").notna().astype(float) * 10
    scored["_mapping_score"] += scored.get("Carehome ID").notna().astype(float) * 10
    scored["_mapping_score"] += (~scored.get("Phase").astype(str).str.contains("Unknown|AREA NOT", case=False, na=False)).astype(float) * 3
    scored["_mapping_score"] += month.rank(method="first").fillna(0) / 1000
    return scored.sort_values("_mapping_score", ascending=False).iloc[0]


def build_caseload_mapping(quarterly_path: Path) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    qdata = load_excel_sheet(quarterly_path, "Data")
    ids = load_excel_sheet(quarterly_path, "IDs")

    mapping: dict[str, dict[str, Any]] = {}
    mapping_rows: list[dict[str, Any]] = []

    qdata = qdata.dropna(subset=["Caseload"]).copy()
    qdata["_caseload_key"] = qdata["Caseload"].map(normalize_name)
    for key, group in qdata.groupby("_caseload_key", dropna=True):
        if not key:
            continue
        row = choose_mapping_row(group)
        record = {
            "mapping_source": "quarterly_data",
            "Caseload": clean_string(row.get("Caseload")),
            "Carehome": clean_string(row.get("Carehome")),
            "Carehome ID": clean_id(row.get("Carehome ID")),
            "Carehome Type": clean_string(row.get("Carehome Type")),
            "CareHome 2 Live List": clean_string(row.get("CareHome 2 Live List")),
            "Training": clean_string(row.get("Training")),
            "Area": clean_string(row.get("Area 2")) or clean_string(row.get("Area")),
            "Phase": clean_string(row.get("Phase")),
            "Post Code": clean_postcode(row.get("Carehome Postcode")),
            "Box ID": clean_string(row.get("Box ID")),
        }
        mapping[key] = record

    ids = ids.dropna(subset=["Caseload"]).copy()
    for _, row in ids.iterrows():
        key = normalize_name(row.get("Caseload"))
        if not key or key in mapping:
            continue
        record = {
            "mapping_source": "ids",
            "Caseload": clean_string(row.get("Caseload")),
            "Carehome": clean_string(row.get("Carehome")),
            "Carehome ID": clean_id(row.get("Carehome id")),
            "Carehome Type": clean_string(row.get("CarehomeType")),
            "CareHome 2 Live List": "",
            "Training": "",
            "Area": clean_string(row.get("Area")),
            "Phase": "",
            "Post Code": "",
            "Box ID": "",
        }
        mapping[key] = record

    for key, record in sorted(mapping.items()):
        mapping_rows.append({"caseload_key": key, **record})

    return mapping, pd.DataFrame(mapping_rows)


def load_observations(observation_path: Path) -> pd.DataFrame:
    parts = []
    workbook = pd.ExcelFile(observation_path)
    for sheet_name in workbook.sheet_names:
        df = pd.read_excel(observation_path, sheet_name=sheet_name, dtype=object)
        df.columns = clean_column_names(df.columns)
        df = df.dropna(how="all").copy()
        df["Source Sheet"] = sheet_name
        parts.append(df)
    observations = pd.concat(parts, ignore_index=True)
    observations["Date/Time"] = combine_date_time(observations["Date"], observations["Time"])
    observations["caseload_key"] = observations["Caseload"].map(normalize_name)
    return observations


ODS_NS = {
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}
ODS_TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"


def ods_cell_text(cell: ET.Element) -> str:
    values = []
    for para in cell.findall(".//text:p", ODS_NS):
        text = "".join(para.itertext()).strip()
        if text:
            values.append(text)
    return " ".join(values).strip()


def ods_row_values(row: ET.Element, max_columns: int | None = None) -> list[str]:
    values: list[str] = []
    for cell in row.findall("table:table-cell", ODS_NS):
        repeat = int(cell.attrib.get(f"{ODS_TABLE}number-columns-repeated", "1"))
        value = ods_cell_text(cell)
        if max_columns is None:
            values.extend([value] * repeat)
            continue
        remaining = max_columns - len(values)
        if remaining <= 0:
            break
        values.extend([value] * min(repeat, remaining))
    return values


def load_ods_locations(ods_path: Path) -> pd.DataFrame:
    required_columns = [
        "Location ID",
        "Care home?",
        "Location Name",
        "Location ODS Code",
        "Care homes beds",
        "Location Local Authority",
        "Location Street Address",
        "Location Address Line 2",
        "Location City",
        "Location County",
        "Location Postal Code",
        "Location Latest Overall Rating",
        "Location Primary Inspection Category",
    ]

    with zipfile.ZipFile(ods_path) as archive:
        with archive.open("content.xml") as content:
            tree = ET.parse(content)

    target_table = None
    for table in tree.getroot().findall(".//table:table", ODS_NS):
        if table.attrib.get(f"{ODS_TABLE}name") == "HSCA_Active_Locations":
            target_table = table
            break
    if target_table is None:
        raise ValueError("HSCA_Active_Locations sheet not found in ODS workbook")

    rows = target_table.findall("table:table-row", ODS_NS)
    header = ods_row_values(rows[0])
    indexes = {col: header.index(col) for col in required_columns if col in header}
    max_col = max(indexes.values()) + 1

    records = []
    for row in rows[1:]:
        values = ods_row_values(row, max_col)
        if not any(values):
            continue
        records.append({col: values[idx] if idx < len(values) else "" for col, idx in indexes.items()})

    locations = pd.DataFrame(records)
    locations["location_name_key"] = locations["Location Name"].map(normalize_name)
    locations["postcode_key"] = locations["Location Postal Code"].map(postcode_key)
    locations["is_care_home"] = locations["Care home?"].astype(str).str.upper().eq("Y")
    locations["is_local_authority"] = locations["Location Local Authority"].astype(str).str.contains(
        LOCAL_AUTHORITY_PATTERN, na=False
    )
    return locations


def best_ods_match(carehome_name: Any, postcode: Any, local_care_homes: pd.DataFrame) -> dict[str, Any]:
    name_key = normalize_name(carehome_name)
    pc_key = postcode_key(postcode)

    if not name_key:
        return {"match_status": "no_match", "score": 0.0}

    if pc_key:
        exact_name_postcode = local_care_homes[
            (local_care_homes["location_name_key"] == name_key) & (local_care_homes["postcode_key"] == pc_key)
        ]
        if not exact_name_postcode.empty:
            return build_ods_match_record(exact_name_postcode.iloc[0], "exact_name_postcode", 1.0)

    exact_name = local_care_homes[local_care_homes["location_name_key"] == name_key]
    if not exact_name.empty:
        return build_ods_match_record(exact_name.iloc[0], "exact_name_only", 1.0)

    if pc_key:
        same_postcode = local_care_homes[local_care_homes["postcode_key"] == pc_key]
        if not same_postcode.empty:
            scored = [
                (difflib.SequenceMatcher(None, name_key, row["location_name_key"]).ratio(), row)
                for _, row in same_postcode.iterrows()
            ]
            score, row = sorted(scored, key=lambda item: item[0], reverse=True)[0]
            status = "same_postcode_candidate" if score >= 0.55 else "same_postcode_low_confidence"
            return build_ods_match_record(row, status, score)

    fuzzy = []
    for _, row in local_care_homes.iterrows():
        score = difflib.SequenceMatcher(None, name_key, row["location_name_key"]).ratio()
        if score >= 0.82:
            fuzzy.append((score, row))
    if fuzzy:
        score, row = sorted(fuzzy, key=lambda item: item[0], reverse=True)[0]
        return build_ods_match_record(row, "fuzzy_name_candidate", score)

    return {"match_status": "no_match", "score": 0.0}


def build_ods_match_record(row: pd.Series, status: str, score: float) -> dict[str, Any]:
    return {
        "match_status": status,
        "score": round(float(score), 3),
        "Location ID": clean_string(row.get("Location ID")),
        "Location Name": clean_string(row.get("Location Name")),
        "Location ODS Code": clean_string(row.get("Location ODS Code")),
        "Care homes beds": to_number(row.get("Care homes beds")),
        "Location Local Authority": clean_string(row.get("Location Local Authority")),
        "Location Street Address": clean_string(row.get("Location Street Address")),
        "Location Address Line 2": clean_string(row.get("Location Address Line 2")),
        "Location City": clean_string(row.get("Location City")),
        "Location County": clean_string(row.get("Location County")),
        "Location Postal Code": clean_postcode(row.get("Location Postal Code")),
        "Location Latest Overall Rating": clean_string(row.get("Location Latest Overall Rating")),
        "Location Primary Inspection Category": clean_string(row.get("Location Primary Inspection Category")),
    }


def build_ods_review(mapped_observations: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    local_care_homes = locations[locations["is_care_home"] & locations["is_local_authority"]].copy()
    carehomes = (
        mapped_observations[
            [
                "mapped_Carehome",
                "mapped_Carehome ID",
                "mapped_Post Code",
                "mapped_Area",
                "mapped_Phase",
            ]
        ]
        .dropna(subset=["mapped_Carehome"])
        .drop_duplicates()
        .copy()
    )

    records = []
    for _, row in carehomes.iterrows():
        match = best_ods_match(row["mapped_Carehome"], row["mapped_Post Code"], local_care_homes)
        records.append(
            {
                "Care Home Name": clean_string(row["mapped_Carehome"]),
                "Care Home ID": clean_id(row["mapped_Carehome ID"]),
                "Post Code": clean_postcode(row["mapped_Post Code"]),
                "Area": clean_string(row["mapped_Area"]),
                "Phase": clean_string(row["mapped_Phase"]),
                **match,
            }
        )
    return pd.DataFrame(records).sort_values(["match_status", "Care Home Name"]).reset_index(drop=True)


def map_observations(observations: pd.DataFrame, mapping: dict[str, dict[str, Any]]) -> pd.DataFrame:
    mapped_records = [mapping.get(key, {}) for key in observations["caseload_key"]]
    mapped = pd.DataFrame(mapped_records).add_prefix("mapped_")
    return pd.concat([observations.reset_index(drop=True), mapped.reset_index(drop=True)], axis=1)


def add_news2_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["O2_New"] = [
        score_oxygen_saturation(o2, delivery, scale)
        for o2, delivery, scale in zip(out["O2"], out["O2 delivery"], out["Scale 2 in use?"], strict=False)
    ]
    out["Systolic_New"] = out["Systolic"].map(score_systolic)
    out["Pulse_New"] = out["Pulse"].map(score_pulse)
    out["Temperate_New"] = out["Temperature"].map(score_temperature)
    out["Respiraties_New"] = out["Respiration rate"].map(score_respiration)
    out["O2 Delivery_New"] = out["O2 delivery"].map(score_o2_delivery)
    out["Consciouness New"] = out["Consciousness"].map(score_consciousness)
    score_cols = [
        "O2_New",
        "Systolic_New",
        "Pulse_New",
        "Temperate_New",
        "Respiraties_New",
        "O2 Delivery_New",
        "Consciouness New",
    ]
    out["Recalculated NEWS2 score"] = out[score_cols].sum(axis=1, min_count=len(score_cols))
    out["Source NEWS2 score"] = pd.to_numeric(out["NEWS2 score"], errors="coerce")
    out["NEWS2 Score Difference"] = out["Recalculated NEWS2 score"] - out["Source NEWS2 score"]
    return out


def build_quality_flags(df: pd.DataFrame) -> pd.Series:
    flags = []
    for _, row in df.iterrows():
        row_flags = []
        caseload = clean_string(row.get("Caseload")).lower()
        if re.search(r"training|test", caseload):
            row_flags.append("training_or_test")
        if caseload.startswith("micare"):
            row_flags.append("micare_unmapped")
        if not clean_string(row.get("mapped_Carehome")) or not clean_string(row.get("mapped_Carehome ID")):
            row_flags.append("unmapped_caseload")
        if clean_id(row.get("mapped_Carehome ID")) in OUT_OF_SCOPE_CAREHOME_IDS:
            row_flags.append("out_of_scope_carehome_id_1082")
        for field in REQUIRED_VITAL_FIELDS:
            value = to_number(row.get(field))
            if pd.isna(value) or value <= 0:
                row_flags.append(f"invalid_{field.lower().replace(' ', '_')}")
        if pd.isna(row.get("Date/Time")):
            row_flags.append("invalid_datetime")
        flags.append(";".join(row_flags))
    return pd.Series(flags, index=df.index)


def lookup_old_metadata(
    row: pd.Series,
    old_by_name: dict[str, dict[str, Any]],
    old_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    name_key = normalize_name(row.get("mapped_Carehome"))
    id_key = clean_id(row.get("mapped_Carehome ID"))
    return old_by_name.get(name_key) or old_by_id.get(id_key) or {}


def add_ods_fields(mapped_df: pd.DataFrame, ods_review: pd.DataFrame) -> pd.DataFrame:
    review = ods_review.copy()
    review["Care Home ID"] = review["Care Home ID"].map(clean_id)
    allowed = {"exact_name_postcode", "exact_name_only"}
    review["auto_apply_ods"] = review["match_status"].isin(allowed)
    review_by_id = review.drop_duplicates("Care Home ID").set_index("Care Home ID").to_dict("index")

    records = []
    for _, row in mapped_df.iterrows():
        record = review_by_id.get(clean_id(row.get("mapped_Carehome ID")), {})
        if not record.get("auto_apply_ods", False):
            record = {**record, "Location ODS Code": "", "Location ID": "", "Care homes beds": np.nan}
        records.append(record)
    return pd.concat([mapped_df.reset_index(drop=True), pd.DataFrame(records).add_prefix("ods_")], axis=1)


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if pd.isna(value):
            continue
        text = clean_string(value)
        if text:
            return value
    return ""


def build_working_data(
    included: pd.DataFrame,
    old_by_name: dict[str, dict[str, Any]],
    old_by_id: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for _, row in included.iterrows():
        old_meta = lookup_old_metadata(row, old_by_name, old_by_id)
        old_or_blank = lambda col: old_meta.get(col, "")

        auto_ods_beds = row.get("ods_Care homes beds")
        old_beds = old_or_blank("No of Beds")
        beds = old_beds if not pd.isna(pd.to_numeric(old_beds, errors="coerce")) else auto_ods_beds

        postal_address = first_non_empty(
            old_or_blank("Postal address"),
            row.get("ods_Location Street Address"),
            row.get("ods_Location Address Line 2"),
        )
        postcode = first_non_empty(old_or_blank("Post Code"), row.get("mapped_Post Code"), row.get("ods_Location Postal Code"))

        output = {
            "Date/Time": row.get("Date/Time"),
            "Care Home ID": clean_id(row.get("mapped_Carehome ID")),
            "Care Home Name": first_non_empty(old_or_blank("Care Home Name"), row.get("mapped_Carehome")),
            "Type": old_or_blank("Type"),
            "Area": first_non_empty(row.get("mapped_Area"), old_or_blank("Area")),
            "Phase": first_non_empty(row.get("mapped_Phase"), old_or_blank("Phase")),
            "Postal address": postal_address,
            "Post Code": clean_postcode(postcode),
            "GP": old_or_blank("GP"),
            "GP Practice": old_or_blank("GP Practice"),
            "GP Postcode": old_or_blank("GP Postcode"),
            "Type of service": first_non_empty(old_or_blank("Type of service"), row.get("mapped_CareHome 2 Live List")),
            "No of Beds": beds,
            "Miles from THA - LE12 8FE": old_or_blank("Miles from THA - LE12 8FE"),
            "Asset number": old_or_blank("Asset number"),
            "Amount of Asset": old_or_blank("Amount of Asset"),
            "Weekly fee": old_or_blank("Weekly fee"),
            "Provider company": old_or_blank("Provider company"),
            "Clinical concern?": row.get("Clinical concern?"),
            "NEWS2 score": row.get("Recalculated NEWS2 score"),
            "New2 Score_New": row.get("Recalculated NEWS2 score"),
            "H=I?": old_or_blank("H=I?"),
            "O2": row.get("O2"),
            "O2_New": row.get("O2_New"),
            "Systolic": row.get("Systolic"),
            "Systolic_New": row.get("Systolic_New"),
            "Diasolic": row.get("Diasolic"),
            "Pulse": row.get("Pulse"),
            "Pulse_New": row.get("Pulse_New"),
            "Temperature": row.get("Temperature"),
            "Temperate_New": row.get("Temperate_New"),
            "Respiration rate": row.get("Respiration rate"),
            "Respiraties_New": row.get("Respiraties_New"),
            "O2 delivery": row.get("O2 delivery"),
            "O2 Delivery_New": row.get("O2 Delivery_New"),
            "Consciousness": row.get("Consciousness"),
            "Consciouness New": row.get("Consciouness New"),
            "Scale 2 in use?": row.get("Scale 2 in use?"),
            "Source Caseload": row.get("Caseload"),
            "Source Sheet": row.get("Source Sheet"),
            "Source NEWS2 score": row.get("Source NEWS2 score"),
            "NEWS2 Score Difference": row.get("NEWS2 Score Difference"),
            "ODS Match Status": row.get("ods_match_status"),
            "ODS Location ID": row.get("ods_Location ID"),
            "ODS Code": row.get("ods_Location ODS Code"),
            "ODS Location Name": row.get("ods_Location Name"),
            "ODS Beds": row.get("ods_Care homes beds"),
            "Data Quality Flags": row.get("quality_flags"),
        }
        rows.append(output)

    working = pd.DataFrame(rows)
    for col in OLD_SCHEMA_COLUMNS + EXTRA_WORKING_COLUMNS:
        if col not in working.columns:
            working[col] = ""
    working = working[OLD_SCHEMA_COLUMNS + EXTRA_WORKING_COLUMNS]
    working["NEWS2 score"] = pd.to_numeric(working["NEWS2 score"], errors="coerce").astype("Int64")
    working["New2 Score_New"] = pd.to_numeric(working["New2 Score_New"], errors="coerce").astype("Int64")
    return working.sort_values(["Date/Time", "Care Home ID", "Source Caseload"]).reset_index(drop=True)


def build_new_care_homes(working: pd.DataFrame, old_df: pd.DataFrame) -> pd.DataFrame:
    old_names = {normalize_name(name) for name in old_df["Care Home Name"].dropna().unique()}
    grouped = (
        working.groupby(["Care Home ID", "Care Home Name", "Area", "Phase"], dropna=False)
        .agg(
            observations=("Date/Time", "size"),
            first_observation=("Date/Time", "min"),
            last_observation=("Date/Time", "max"),
            postcode=("Post Code", "first"),
            beds=("No of Beds", "first"),
        )
        .reset_index()
    )
    grouped["matched_old_working_data"] = grouped["Care Home Name"].map(lambda name: normalize_name(name) in old_names)
    return grouped.sort_values(["matched_old_working_data", "Care Home Name"]).reset_index(drop=True)


def build_duplicate_id_review(working: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        working.groupby(["Care Home ID", "Care Home Name"], dropna=False)
        .agg(
            observations=("Date/Time", "size"),
            source_caseloads=("Source Caseload", lambda values: "; ".join(sorted(set(map(str, values))))),
            first_observation=("Date/Time", "min"),
            last_observation=("Date/Time", "max"),
            postcode=("Post Code", "first"),
        )
        .reset_index()
    )
    duplicate_ids = (
        grouped.groupby("Care Home ID")["Care Home Name"]
        .nunique()
        .reset_index(name="care_home_name_count")
        .query("care_home_name_count > 1")
    )
    if duplicate_ids.empty:
        return pd.DataFrame(
            columns=[
                "Care Home ID",
                "care_home_name_count",
                "Care Home Name",
                "observations",
                "source_caseloads",
                "first_observation",
                "last_observation",
                "postcode",
            ]
        )
    return duplicate_ids.merge(grouped, on="Care Home ID", how="left").sort_values(
        ["Care Home ID", "Care Home Name"]
    )


def build_monthly_validation(raw_scored: pd.DataFrame, working: pd.DataFrame, quarterly_path: Path) -> pd.DataFrame:
    qdata = load_excel_sheet(quarterly_path, "Data")
    qdata["month"] = pd.to_datetime(qdata["Month"], errors="coerce").dt.to_period("M").astype(str)
    q_total = pd.to_numeric(qdata.get("NEWS2 (Total)"), errors="coerce").fillna(0).groupby(qdata["month"]).sum()

    raw = raw_scored.copy()
    raw["month"] = raw["Date/Time"].dt.to_period("M").astype(str)
    raw_counts = raw.groupby("month").size()

    primary = working.copy()
    primary["month"] = pd.to_datetime(primary["Date/Time"], errors="coerce").dt.to_period("M").astype(str)
    primary_counts = primary.groupby("month").size()

    all_months = sorted(set(q_total.index) | set(raw_counts.index) | set(primary_counts.index))
    validation = pd.DataFrame({"month": all_months})
    validation["quarterly_NEWS2_total"] = validation["month"].map(q_total).fillna(0).astype(int)
    validation["raw_escalation_rows"] = validation["month"].map(raw_counts).fillna(0).astype(int)
    validation["primary_working_rows"] = validation["month"].map(primary_counts).fillna(0).astype(int)
    validation["raw_minus_quarterly"] = validation["raw_escalation_rows"] - validation["quarterly_NEWS2_total"]
    validation["primary_minus_quarterly"] = validation["primary_working_rows"] - validation["quarterly_NEWS2_total"]
    return validation


def build_actual_news_inventory(phase2_raw_dir: Path) -> pd.DataFrame:
    files = []
    for path in sorted(phase2_raw_dir.rglob("*.xlsx")):
        if path.name in {"Whzan LLR Quarterly March 2026.xlsx", "rutland leicester news2 summary data, 05 may 2026.xlsx"}:
            continue
        try:
            workbook = pd.ExcelFile(path)
            sheets = ", ".join(workbook.sheet_names)
        except Exception as exc:  # pragma: no cover - inventory should not stop processing.
            sheets = f"ERROR: {exc}"
        files.append(
            {
                "file": str(path.relative_to(phase2_raw_dir)),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
                "sheets": sheets,
            }
        )
    return pd.DataFrame(files)


def validate_working_data(working: pd.DataFrame) -> list[str]:
    errors = []
    required = [
        "Date/Time",
        "Care Home ID",
        "Care Home Name",
        "Area",
        "Phase",
        "Post Code",
        "No of Beds",
        "NEWS2 score",
        "O2_New",
        "Systolic_New",
        "Pulse_New",
        "Temperate_New",
        "Respiraties_New",
        "O2 Delivery_New",
        "Consciouness New",
    ]
    missing_columns = [col for col in required if col not in working.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")

    if working["Care Home ID"].map(clean_string).eq("").any():
        errors.append("Care Home ID contains blanks")
    if pd.to_datetime(working["Date/Time"], errors="coerce").isna().any():
        errors.append("Date/Time contains invalid values")
    if pd.to_numeric(working["NEWS2 score"], errors="coerce").isna().any():
        errors.append("NEWS2 score contains invalid values")
    if working["Source Caseload"].astype(str).str.contains("training|test|^Micare", case=False, na=False).any():
        errors.append("Primary working data still contains training/test/MiCare caseloads")

    for field in REQUIRED_VITAL_FIELDS:
        values = pd.to_numeric(working[field], errors="coerce")
        if values.isna().any() or (values <= 0).any():
            errors.append(f"{field} contains missing or non-positive values")
    return errors


def write_markdown_summary(
    path: Path,
    inputs: dict[str, Path],
    working: pd.DataFrame,
    raw_scored: pd.DataFrame,
    excluded: pd.DataFrame,
    ods_review: pd.DataFrame,
    new_care_homes: pd.DataFrame,
    validation_errors: list[str],
) -> None:
    source_diff = working["NEWS2 Score Difference"].dropna()
    changed_scores = int((source_diff != 0).sum())
    exact_ods = ods_review["match_status"].value_counts().to_dict()
    new_only = new_care_homes[~new_care_homes["matched_old_working_data"]]
    duplicate_ids = (
        new_care_homes.groupby("Care Home ID")["Care Home Name"]
        .nunique()
        .reset_index(name="care_home_name_count")
        .query("care_home_name_count > 1")
    )

    lines = [
        "# Phase 2 Data Processing Summary",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Inputs",
    ]
    for label, input_path in inputs.items():
        lines.append(f"- {label}: `{input_path}`")

    lines.extend(
        [
            "",
            "## Output Summary",
            f"- Raw observation rows: {len(raw_scored):,}",
            f"- Primary working rows: {len(working):,}",
            f"- Excluded/audit rows: {len(excluded):,}",
            f"- Care homes in primary working data: {working['Care Home ID'].nunique():,}",
            f"- Date range: {working['Date/Time'].min()} to {working['Date/Time'].max()}",
            f"- Included rows where recalculated NEWS2 differs from source score: {changed_scores:,}",
            f"- Care Home IDs mapped to more than one name: {len(duplicate_ids):,}",
            "",
            "## ODS Matching",
        ]
    )
    for status, count in sorted(exact_ods.items()):
        lines.append(f"- {status}: {count}")

    lines.extend(
        [
            "",
            "## New Or Unmatched Care Homes",
            f"- Count not matched to old working data by normalized name: {len(new_only):,}",
        ]
    )
    for name in new_only["Care Home Name"].head(30):
        lines.append(f"- {name}")

    lines.extend(
        [
            "",
            "## Validation",
        ]
    )
    if validation_errors:
        lines.extend(f"- ERROR: {error}" for error in validation_errors)
    else:
        lines.append("- Passed local schema and quality checks.")

    lines.extend(
        [
            "",
            "## Rules Applied",
            "- Training/test caseloads excluded from the primary working data.",
            "- MiCare caseloads excluded because they are not mapped to a care home in the Phase 2 reference workbook.",
            "- Carehome ID 1082 is excluded from scope because the source maps it to both Curtis Weston House and Sanctuary Supported Living.",
            "- Rows with missing or non-positive required vitals excluded from the primary working data.",
            "- NEWS2 score in the primary workbook is recalculated from raw vitals using standard NEWS2 rules.",
            "- Source NEWS2 score is retained for audit.",
            "- Only exact ODS name+postcode and exact ODS name matches are applied automatically.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    paths: OutputPaths,
    working: pd.DataFrame,
    audit_sheets: dict[str, pd.DataFrame],
    summary_inputs: dict[str, Path],
    raw_scored: pd.DataFrame,
    excluded: pd.DataFrame,
    ods_review: pd.DataFrame,
    new_care_homes: pd.DataFrame,
    validation_errors: list[str],
) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(paths.working_data, engine="openpyxl") as writer:
        working.to_excel(writer, sheet_name="working_data", index=False)

    with pd.ExcelWriter(paths.audit_report, engine="openpyxl") as writer:
        for sheet_name, df in audit_sheets.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

    write_markdown_summary(
        paths.summary_markdown,
        summary_inputs,
        working,
        raw_scored,
        excluded,
        ods_review,
        new_care_homes,
        validation_errors,
    )


def prepare_phase2_data(args: argparse.Namespace) -> dict[str, Any]:
    old_df = pd.read_excel(args.old_working_data, dtype=object)
    old_df.columns = clean_column_names(old_df.columns)
    old_by_name, old_by_id = build_old_metadata(old_df)

    mapping, mapping_review = build_caseload_mapping(args.quarterly_workbook)
    observations = load_observations(args.observation_workbook)
    mapped = map_observations(observations, mapping)
    scored = add_news2_scores(mapped)
    scored["quality_flags"] = build_quality_flags(scored)

    locations = load_ods_locations(args.ods_workbook)
    ods_review = build_ods_review(scored, locations)
    enriched = add_ods_fields(scored, ods_review)
    enriched["quality_flags"] = scored["quality_flags"]

    included_mask = enriched["quality_flags"].eq("")
    included = enriched[included_mask].copy()
    excluded = enriched[~included_mask].copy()

    working = build_working_data(included, old_by_name, old_by_id)
    validation_errors = validate_working_data(working)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    excluded_export_cols = [
        "Date",
        "Time",
        "Date/Time",
        "Caseload",
        "Source Sheet",
        "quality_flags",
        "Clinical concern?",
        "NEWS2 score",
        "Recalculated NEWS2 score",
        "Source NEWS2 score",
        "NEWS2 Score Difference",
        "O2",
        "Systolic",
        "Diasolic",
        "Pulse",
        "Temperature",
        "Respiration rate",
        "O2 delivery",
        "Consciousness",
        "Scale 2 in use?",
        "mapped_Carehome",
        "mapped_Carehome ID",
        "mapped_Area",
        "mapped_Phase",
        "mapped_Post Code",
    ]
    excluded_export_cols = [col for col in excluded_export_cols if col in excluded.columns]

    score_diff_cols = [
        "Date/Time",
        "Caseload",
        "Source Sheet",
        "Source NEWS2 score",
        "Recalculated NEWS2 score",
        "NEWS2 Score Difference",
        "O2",
        "Systolic",
        "Pulse",
        "Temperature",
        "Respiration rate",
        "O2 delivery",
        "Consciousness",
        "quality_flags",
    ]
    score_diff = enriched[
        pd.to_numeric(enriched["NEWS2 Score Difference"], errors="coerce").fillna(0).ne(0)
    ][[col for col in score_diff_cols if col in enriched.columns]].copy()

    new_care_homes = build_new_care_homes(working, old_df)
    duplicate_id_review = build_duplicate_id_review(working)
    monthly_validation = build_monthly_validation(scored, working, args.quarterly_workbook)
    actual_news_inventory = build_actual_news_inventory(args.phase2_raw_dir)

    summary_metrics = pd.DataFrame(
        [
            ("raw_observation_rows", len(scored)),
            ("primary_working_rows", len(working)),
            ("excluded_rows", len(excluded)),
            ("care_home_count", working["Care Home ID"].nunique()),
            ("date_min", working["Date/Time"].min()),
            ("date_max", working["Date/Time"].max()),
            ("source_score_differences_in_primary", int((working["NEWS2 Score Difference"].fillna(0) != 0).sum())),
            ("care_home_ids_with_multiple_names", len(duplicate_id_review["Care Home ID"].unique())),
            ("ods_exact_name_postcode", int((ods_review["match_status"] == "exact_name_postcode").sum())),
            ("ods_exact_name_only", int((ods_review["match_status"] == "exact_name_only").sum())),
            (
                "ods_candidate_or_unmatched",
                int((~ods_review["match_status"].isin(["exact_name_postcode", "exact_name_only"])).sum()),
            ),
        ],
        columns=["metric", "value"],
    )

    audit_sheets = {
        "summary_metrics": summary_metrics,
        "excluded_rows": excluded[excluded_export_cols].sort_values(["quality_flags", "Date/Time", "Caseload"]),
        "caseload_mapping": mapping_review,
        "ods_match_review": ods_review,
        "new_care_homes": new_care_homes,
        "duplicate_ids": duplicate_id_review,
        "monthly_validation": monthly_validation,
        "score_differences": score_diff.sort_values(["quality_flags", "Date/Time", "Caseload"]),
        "actual_news_files": actual_news_inventory,
    }

    paths = OutputPaths(
        output_dir=args.output_dir,
        working_data=args.output_dir / "phase2_working_data_recalculated_news2.xlsx",
        audit_report=args.output_dir / "phase2_audit_report.xlsx",
        summary_markdown=args.output_dir / "phase2_data_processing_summary.md",
    )
    write_outputs(
        paths,
        working,
        audit_sheets,
        {
            "old_working_data": args.old_working_data,
            "observation_workbook": args.observation_workbook,
            "quarterly_workbook": args.quarterly_workbook,
            "ods_workbook": args.ods_workbook,
        },
        scored,
        excluded,
        ods_review,
        new_care_homes,
        validation_errors,
    )

    return {
        "paths": paths,
        "working": working,
        "excluded": excluded,
        "ods_review": ods_review,
        "new_care_homes": new_care_homes,
        "monthly_validation": monthly_validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 2 working data for the Carehome dashboard.")
    parser.add_argument("--phase2-raw-dir", type=Path, default=PHASE2_RAW_DIR)
    parser.add_argument("--old-working-data", type=Path, default=DEFAULT_OLD_WORKING_DATA)
    parser.add_argument("--observation-workbook", type=Path, default=DEFAULT_OBSERVATION_WORKBOOK)
    parser.add_argument("--quarterly-workbook", type=Path, default=DEFAULT_QUARTERLY_WORKBOOK)
    parser.add_argument("--ods-workbook", type=Path, default=DEFAULT_ODS_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_phase2_data(args)
    paths: OutputPaths = result["paths"]
    working: pd.DataFrame = result["working"]
    excluded: pd.DataFrame = result["excluded"]
    ods_review: pd.DataFrame = result["ods_review"]
    new_care_homes: pd.DataFrame = result["new_care_homes"]

    print(f"Working data: {paths.working_data}")
    print(f"Audit report: {paths.audit_report}")
    print(f"Summary: {paths.summary_markdown}")
    print(f"Rows: working={len(working):,}, excluded={len(excluded):,}")
    print(f"Care homes: {working['Care Home ID'].nunique():,}")
    print(f"Date range: {working['Date/Time'].min()} to {working['Date/Time'].max()}")
    print(f"ODS match counts: {ods_review['match_status'].value_counts().to_dict()}")
    print(
        "New/unmatched care homes: "
        f"{int((~new_care_homes['matched_old_working_data']).sum()):,}"
    )


if __name__ == "__main__":
    main()
