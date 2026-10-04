"""
JoSAA Cutoff Lookup Module
Primary Responsibility: Answers rank-based questions ("what can I get with rank 12000?",
"NIT Trichy CSE closing rank") from the structured JoSAA opening/closing rank table.
Why it exists: 465k cutoff rows cannot be chunked into FAISS usefully, so they are
filtered directly and handed to the LLM as one compact, exact context block.
"""
import os
import re
import pandas as pd

from src.agentops.telemetry import tracer

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "data", "structured", "josaa_cutoffs_2018_2025.csv.gz")
SOURCE = "JoSAA Opening & Closing Ranks 2018-2025 (structured lookup)"
MAX_ROWS = 25

CATEGORY_PATTERNS = [
    ("OBC-NCL", r"\bobc\b"),
    ("EWS", r"\bews\b"),
    ("SC", r"\bsc\b"),
    ("ST", r"\bst\b"),
]

BRANCH_PATTERNS = {
    "Computer Science": r"\bcse\b|computer|\bcs\b",
    "Electronics": r"\bece\b|electronics",
    "Electrical": r"\beee\b|\bee\b|electrical",
    "Mechanical": r"mechanical|\bmech\b",
    "Civil": r"\bcivil\b",
    "Chemical": r"chemical",
    "Artificial Intelligence": r"\bai\b|artificial intelligence|machine learning",
    "Data Science": r"data science",
    "Mathematics": r"math",
}

# Common short names -> words that appear in the official JoSAA institute name
INSTITUTE_ALIASES = {
    "trichy": "Tiruchirappalli", "surathkal": "Karnataka, Surathkal", "nitk": "Karnataka, Surathkal",
    "bombay": "Bombay", "mumbai": "Bombay", "madras": "Madras", "chennai": "Madras",
    "delhi": "Delhi", "kanpur": "Kanpur", "kharagpur": "Kharagpur", "kgp": "Kharagpur",
    "roorkee": "Roorkee", "guwahati": "Guwahati", "hyderabad": "Hyderabad", "bhu": "(BHU) Varanasi",
    "warangal": "Warangal", "rourkela": "Rourkela", "calicut": "Calicut", "allahabad": "Allahabad",
    "mnnit": "Allahabad", "jaipur": "Jaipur", "mnit": "Jaipur", "bhopal": "Bhopal", "manit": "Bhopal",
    "nagpur": "Nagpur", "vnit": "Nagpur", "surat": "Surat", "kurukshetra": "Kurukshetra",
    "durgapur": "Durgapur", "silchar": "Silchar", "jamshedpur": "Jamshedpur", "patna": "Patna",
    "raipur": "Raipur", "hamirpur": "Hamirpur", "jalandhar": "Jalandhar", "srinagar": "Srinagar",
    "agartala": "Agartala", "goa": "Goa", "puducherry": "Puducherry", "dhanbad": "Dhanbad",
    "ism": "Dhanbad", "indore": "Indore", "gandhinagar": "Gandhinagar", "ropar": "Ropar",
    "mandi": "Mandi", "jodhpur": "Jodhpur", "patna iit": "Patna", "tirupati": "Tirupati",
    "palakkad": "Palakkad", "bhilai": "Bhilai", "dharwad": "Dharwad", "jammu": "Jammu",
    "bhubaneswar": "Bhubaneswar", "gwalior": "Gwalior", "jabalpur": "Jabalpur", "lucknow": "Lucknow",
    "kota": "Kota", "sricity": "Sri City", "sri city": "Sri City",
}

TYPE_PATTERNS = [
    ("IIIT", r"\biiits?\b"),
    ("IIT", r"\biits?\b"),
    ("NIT", r"\bnits?\b"),
]


def _to_rank(series: pd.Series) -> pd.Series:
    # "P" suffix marks preparatory-course ranks (PwD seats); they are not comparable to normal ranks
    return pd.to_numeric(series.where(~series.str.endswith("P", na=False)), errors="coerce")


def _institute_type(name: str) -> str:
    if name.startswith("Indian Institute of Technology"):
        return "IIT"
    if name.startswith("National Institute of Technology") or "Malaviya National" in name \
            or "Maulana Azad National" in name or "Visvesvaraya National" in name \
            or "Sardar Vallabhbhai National" in name or "Motilal Nehru National" in name:
        return "NIT"
    if name.startswith("Indian Institute of Information Technology") or name.startswith("Atal Bihari Vajpayee") \
            or "Indian Institute of Information Technology" in name:
        return "IIIT"
    return "GFTI"


class CutoffLookup:
    def __init__(self, csv_path: str = CSV_PATH):
        self.available = os.path.exists(csv_path)
        if not self.available:
            return
        df = pd.read_csv(csv_path, dtype=str).dropna(subset=["institute", "academic program name"])
        df["year"] = df["year"].astype(int)
        df["round"] = df["round"].astype(int)
        df["opening"] = _to_rank(df["opening rank"])
        df["closing"] = _to_rank(df["closing rank"])
        df = df.dropna(subset=["closing"])
        df["itype"] = df["institute"].map(_institute_type)
        self.df = df
        self.latest_year = int(df["year"].max())

    def _parse(self, query: str) -> dict:
        q = query.lower()
        rank = None
        matches = list(re.finditer(r"(?:rank|air|crl)\D{0,15}?(\d[\d,]*\.?\d*)\s*(k|lakh|lac)?\b", q)) \
            + list(re.finditer(r"(\d[\d,]*\.?\d*)\s*(k|lakh|lac)?\s*(?:rank|air|crl)", q))
        for m in matches:
            num = float(m.group(1).replace(",", ""))
            mult = {"k": 1_000, "lakh": 100_000, "lac": 100_000}.get(m.group(2) or "", 1)
            if mult == 1 and 2017 <= num <= 2030:
                continue  # "closing rank 2024" is a year, not a rank
            rank = int(num * mult)
            break

        category = next((c for c, p in CATEGORY_PATTERNS if re.search(p, q)), "OPEN")
        if re.search(r"pwd|disab", q):
            category += " (PwD)"

        years = [int(y) for y in re.findall(r"\b(20(?:1[7-9]|2[0-9]))\b", q)]
        year = years[0] if years and years[0] in set(self.df["year"]) else self.latest_year

        branches = [b for b, p in BRANCH_PATTERNS.items() if re.search(p, q)]
        itypes = [t for t, p in TYPE_PATTERNS if re.search(p, q)]
        if "advanced" in q and not itypes:
            itypes = ["IIT"]
        institutes = sorted({v for k, v in INSTITUTE_ALIASES.items() if re.search(rf"\b{re.escape(k)}\b", q)})

        return {
            "rank": rank,
            "category": category,
            "female": bool(re.search(r"\bfemale\b|\bgirl\b|\bwoman\b", q)),
            "year": year,
            "branches": branches,
            "itypes": itypes,
            "institutes": institutes,
        }

    def is_relevant(self, query: str) -> bool:
        q = query.lower()
        if not re.search(r"\brank\b|\bair\b|\bcrl\b|cutoff|cut-off|cut off|closing|opening|which college|can i get", q):
            return False
        # Needs a rank, or a specific institute/branch/type to look up; "percentile cutoff" alone is not a JoSAA rank question
        p = self._parse(query)
        return p["rank"] is not None or bool(p["institutes"] or p["itypes"] or p["branches"])

    def lookup(self, query: str):
        if not self.available or not self.is_relevant(query):
            return None

        with tracer.start_as_current_span("cutoff.lookup") as span:
            p = self._parse(query)
            span.set_attribute("parsed_rank", p["rank"] or -1)
            span.set_attribute("parsed_category", p["category"])
            span.set_attribute("parsed_year", p["year"])

            d = self.df[(self.df["year"] == p["year"]) & (self.df["seat type"] == p["category"])]
            d = d[d["round"] == d["round"].max()]  # final round = the real closing cutoff
            genders = ["Gender-Neutral", "Female-only (including Supernumerary)"] if p["female"] else ["Gender-Neutral"]
            d = d[d["gender"].isin(genders)]
            d = d[d["quota"].isin(["AI", "OS"])]  # home-state quota needs the student's state

            if p["itypes"]:
                d = d[d["itype"].isin(p["itypes"])]
            if p["institutes"]:
                d = d[d["institute"].apply(lambda n: any(a in n for a in p["institutes"]))]
            if p["branches"]:
                pat = "|".join(re.escape(b) for b in p["branches"])
                d = d[d["academic program name"].str.contains(pat, case=False, regex=True)]

            if p["rank"] is not None:
                highest = d["closing"].max()
                d = d[d["closing"] >= p["rank"]].sort_values("closing")
                if d.empty and pd.notna(highest):
                    return {
                        "content": (f"No JoSAA program in the {p['year']} final round (category {p['category']}, "
                                    f"AI/OS quota, filters applied) closed at or above rank {p['rank']:,}. The highest "
                                    f"closing rank among the matching programs was {int(highest):,}. Home-state quota, "
                                    f"CSAB special rounds, or state counselling may still be options."),
                        "metadata": {"source": SOURCE, "year": p["year"], "category": p["category"]},
                        "distance": 0.0,
                    }
                header = (f"Programs whose final-round {p['year']} closing rank was at or above rank {p['rank']:,} "
                          f"(category {p['category']}{', incl. female-only seats' if p['female'] else ''}), "
                          f"most competitive first:")
            else:
                d = d.sort_values("closing")
                header = f"Final-round {p['year']} opening/closing ranks (category {p['category']}):"

            span.set_attribute("rows_matched", len(d))
            if d.empty:
                return None

            lines = [header]
            for _, r in d.head(MAX_ROWS).iterrows():
                gender = " [female-only]" if r["gender"].startswith("Female") else ""
                lines.append(f"- {r['institute']} | {r['academic program name']} | quota {r['quota']}{gender} | "
                             f"opening {int(r['opening']) if pd.notna(r['opening']) else 'n/a'} | closing {int(r['closing'])}")
            if len(d) > MAX_ROWS:
                lines.append(f"... and {len(d) - MAX_ROWS} more matching programs.")
            lines.append("Notes: IIT cutoffs are JEE (Advanced) category ranks; NIT/IIIT/GFTI cutoffs are JEE (Main) ranks. "
                         "Only All-India (AI) and Other-State (OS) quota rows are shown; home-state (HS) cutoffs at NITs are "
                         "usually easier. Past cutoffs are a guide, not a guarantee for the current year.")

            return {
                "content": "\n".join(lines),
                "metadata": {"source": SOURCE, "year": p["year"], "category": p["category"]},
                "distance": 0.0,
            }
