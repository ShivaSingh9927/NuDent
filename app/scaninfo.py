"""Parse the `.scanInfo` XML file shipped with each case folder.

The file lists every tooth in the scan with its FDI number and a
`ReconstructionType` tag. We use this to (a) auto-pick which FDI the user is
designing and (b) decide which arch is the prep vs the opposing.

ReconstructionType values seen in the sample data:
- Antagonist     → tooth on the opposite arch (occlusion reference)
- HealthyTooth   → untouched neighbour
- Coping, Crown, Inlay, Onlay, Veneer, Bridge, ... → PREP (what we design)
"""
import os
import xml.etree.ElementTree as ET


# Anything not in this set is treated as a prep (i.e. something we design for).
NON_PREP_TYPES = {"Antagonist", "HealthyTooth", "Missing"}


def find_scaninfo(folder):
    """Return path to the first *.scanInfo file in `folder`, or None."""
    for name in os.listdir(folder):
        if name.lower().endswith(".scaninfo"):
            return os.path.join(folder, name)
    return None


def parse_scaninfo(path):
    """Parse a `.scanInfo` XML file and return a dict:

        {
            "preps":       [(fdi:int, type:str), ...],
            "antagonists": [int, ...],
            "healthy":     [int, ...],
            "all":         [(fdi:int, type:str), ...],
        }

    Tags use the default XML namespace (none) in the sample files, so we read
    them directly. Unknown FDIs (non-integer or out of range) are skipped.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    preps, antagonists, healthy, all_teeth = [], [], [], []
    for tooth in root.iter("ScanTooth"):
        n_elem = tooth.find("Number")
        t_elem = tooth.find("ReconstructionType")
        if n_elem is None or t_elem is None:
            continue
        try:
            fdi = int((n_elem.text or "").strip())
        except ValueError:
            continue
        if not (11 <= fdi <= 48):
            continue
        rtype = (t_elem.text or "").strip()
        all_teeth.append((fdi, rtype))
        if rtype == "Antagonist":
            antagonists.append(fdi)
        elif rtype == "HealthyTooth":
            healthy.append(fdi)
        elif rtype not in NON_PREP_TYPES:
            preps.append((fdi, rtype))

    return {
        "preps": preps,
        "antagonists": antagonists,
        "healthy": healthy,
        "all": all_teeth,
    }


def prep_arch(prep_fdis):
    """Return 'upper', 'lower', or 'mixed' based on a list of prep FDIs.

    FDI quadrants 1/2 are upper (11-28); 3/4 are lower (31-48).
    """
    if not prep_fdis:
        return None
    uppers = [f for f in prep_fdis if 11 <= f <= 28]
    lowers = [f for f in prep_fdis if 31 <= f <= 48]
    if uppers and not lowers:
        return "upper"
    if lowers and not uppers:
        return "lower"
    return "mixed"
