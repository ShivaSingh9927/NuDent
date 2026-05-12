"""FDI tooth-number ↔ library-file mapping.

FDI quadrants (from the patient's perspective):
    1 = upper right     2 = upper left
    4 = lower right     3 = lower left

Position digit (1..8): 1=central incisor, 2=lateral incisor, 3=canine,
4=first premolar, 5=second premolar, 6=first molar, 7=second molar,
8=third molar.

Library convention: anatomies are stored as the LEFT-side variant. This
matches the explicit `*-left-*` filenames we already have and treats the
non-side-specified files (e.g. `maxillary-canine.stl`) as left too. Teeth
in the right-side quadrants (1 and 4) need to be mirrored across the
mid-sagittal plane (negate X) when loaded.
"""
from typing import Optional


# Position digit → anatomy name. Used for UI labels.
POSITION_NAMES = {
    1: "central incisor",
    2: "lateral incisor",
    3: "canine",
    4: "first premolar",
    5: "second premolar",
    6: "first molar",
    7: "second molar",
    8: "third molar",
}

# (jaw, position) → ordered list of library basenames to try (no .stl).
# We try each candidate in order; the first one that exists in the library
# directory wins. This lets us tolerate the existing inconsistent naming
# (some files say `-left-`, some don't) without renaming any files.
LIBRARY_CANDIDATES = {
    # ---- Maxillary ----
    ("maxillary",  1): ["maxillary-left-central-incisor",
                        "maxillary-central-incisor"],
    ("maxillary",  2): ["maxillary-lateral-incisor",
                        "maxillary-left-lateral-incisor"],
    ("maxillary",  3): ["maxillary-canine",
                        "maxillary-left-canine"],
    ("maxillary",  4): ["maxillary-first-premolar",
                        "maxillary-left-first-premolar"],
    ("maxillary",  5): ["maxillary-second-premolar",
                        "maxillary-left-second-premolar"],
    ("maxillary",  6): ["maxillary-first-molar",
                        "maxillary-first-molar-with-cusp-of-carabelli-0"],
    ("maxillary",  7): ["maxillary-second-molar"],
    ("maxillary",  8): ["maxillary-third-molar"],
    # ---- Mandibular ----
    ("mandibular", 1): ["mandibular-central-incisor",
                        "mandibular-left-central-incisor"],
    ("mandibular", 2): ["mandibular-lateral-incisor",
                        "mandibular-left-lateral-incisor"],
    ("mandibular", 3): ["mandibular-left-canine",
                        "mandibular-canine"],
    ("mandibular", 4): ["mandibular-first-premolar",
                        "mandibular-left-first-premolar"],
    ("mandibular", 5): ["mandibular-left-second-premolar",
                        "mandibular-second-premolar"],
    ("mandibular", 6): ["mandibular-first-molar"],
    ("mandibular", 7): ["mandibular-second-molar"],
    ("mandibular", 8): ["mandibular-third-molar"],
}


def split_fdi(fdi: int):
    """Return (quadrant, position) for a valid FDI number, else (None, None)."""
    if not isinstance(fdi, int) or not (11 <= fdi <= 48):
        return None, None
    q, p = divmod(fdi, 10)
    if q < 1 or q > 4 or p < 1 or p > 8:
        return None, None
    return q, p


def jaw_of(fdi: int) -> Optional[str]:
    q, _ = split_fdi(fdi)
    if q is None:
        return None
    return "maxillary" if q in (1, 2) else "mandibular"


def side_of(fdi: int) -> Optional[str]:
    """Patient's left or right."""
    q, _ = split_fdi(fdi)
    if q is None:
        return None
    return "left" if q in (2, 3) else "right"


def needs_mirror(fdi: int) -> bool:
    """Library is stored left-side, so right quadrants need a sagittal mirror."""
    return side_of(fdi) == "right"


def tooth_name(fdi: int) -> str:
    """Human-readable description like 'maxillary right canine'."""
    q, p = split_fdi(fdi)
    if q is None:
        return f"tooth {fdi}"
    return f"{jaw_of(fdi)} {side_of(fdi)} {POSITION_NAMES[p]}"


def library_file_for(fdi: int, available_files) -> Optional[str]:
    """Return the best-matching library filename (with .stl), or None.

    `available_files` should be the set/list of filenames present in the
    library directory.
    """
    available = set(available_files)
    q, p = split_fdi(fdi)
    if q is None:
        return None
    candidates = LIBRARY_CANDIDATES.get((jaw_of(fdi), p), [])
    for stem in candidates:
        fname = f"{stem}.stl"
        if fname in available:
            return fname
    return None


def resolve(fdi: int, available_files) -> dict:
    """One-shot lookup. Returns a dict ready for downstream use.

    Keys:
        file     : library filename (or None if no match)
        mirror   : bool — flip X when loading (right-quadrant compensation)
        name     : human-readable description
        jaw      : 'maxillary' | 'mandibular' | None
        side     : 'left' | 'right' | None
        position : anatomy name (e.g. 'first molar') | None
        fdi      : the original number
    """
    q, p = split_fdi(fdi)
    if q is None:
        return {
            "file": None, "mirror": False, "name": f"tooth {fdi}",
            "jaw": None, "side": None, "position": None, "fdi": fdi,
        }
    return {
        "file": library_file_for(fdi, available_files),
        "mirror": needs_mirror(fdi),
        "name": tooth_name(fdi),
        "jaw": jaw_of(fdi),
        "side": side_of(fdi),
        "position": POSITION_NAMES[p],
        "fdi": fdi,
    }


def coverage_report(available_files) -> list:
    """Return [(fdi, status), ...] for all 32 teeth.

    status ∈ {'exact', 'mirror', 'missing'}.
    Useful for a one-time audit of the library + for the tooth-chart UI
    to grey out teeth we can't auto-place.
    """
    report = []
    for q in range(1, 5):
        for p in range(1, 9):
            fdi = q * 10 + p
            f = library_file_for(fdi, available_files)
            if f is None:
                status = "missing"
            elif needs_mirror(fdi):
                status = "mirror"
            else:
                status = "exact"
            report.append((fdi, status))
    return report
