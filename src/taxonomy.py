"""Phase 2: single source of truth for item type -> waste category.

Source for the four categories and their examples:
Instruksi Gubernur DKI Jakarta No. 5 Tahun 2026, Lampiran A
https://jdih.jakarta.go.id/link/download-fulltext/14954
This is a Jakarta regional instruction; other regions may differ.

Run  python -m src.taxonomy  to print the table.
"""
import re

CATEGORIES = ["organik", "anorganik", "b3", "residu"]
CATEGORY_NAMES = {
    "organik": "Sampah Organik",
    "anorganik": "Sampah Anorganik",
    "b3": "Sampah B3",
    "residu": "Sampah Residu",
}

# Keys are normalised item names (lowercase, non-alphanumerics -> "_").
# value = (category, basis)
ITEM_TO_CATEGORY = {
    "food_waste": ("organik", "Lampiran A: sisa makanan / sisa memasak"),
    "garden_waste": ("organik", "Lampiran A: daun"),
    "other_organic_waste": ("organik", "Lampiran A: sampah mudah terurai lainnya"),
    "paper": ("anorganik", "Lampiran A: kertas, kardus"),
    "plastic": ("anorganik", "Lampiran A: botol/kemasan plastik"),
    "glass": ("anorganik", "Lampiran A: botol kaca"),
    "metal": ("anorganik", "Lampiran A: logam"),
    "battery": ("b3", "Lampiran A: baterai"),
    "e_waste": ("b3", "Lampiran A: e-waste"),
    "trash": ("residu", "Lampiran A: sampah yang tertolak pengolahan lanjut"),
    # TEAM DECISION: not listed in Lampiran A. Default = residu (conservative:
    # a photo cannot show whether a local offtaker accepts it). Change here if
    # the team decides otherwise, and write the reason in the report.
    "textile_fabric": ("residu", "Keputusan tim: tidak disebut di Lampiran A"),
}

# Alternative spellings that may appear as folder names. Add only after seeing
# the real folder names printed by inspect_nextech / check_mapping.
ALIASES = {
    "ewaste": "e_waste",
    "general_trash": "trash",
    "textile": "textile_fabric",
    "other_organic": "other_organic_waste",
}


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def item_key(name: str) -> str:
    key = normalize(name)
    return ALIASES.get(key, key)


def category_of(item_name: str) -> str:
    key = item_key(item_name)
    if key not in ITEM_TO_CATEGORY:
        raise KeyError(f"No category defined for item '{item_name}' (key '{key}'). "
                       "Add it to ITEM_TO_CATEGORY or ALIASES in src/taxonomy.py.")
    return ITEM_TO_CATEGORY[key][0]


def main():
    print(f"{'item key':<22}{'category':<12}basis")
    for key, (cat, basis) in ITEM_TO_CATEGORY.items():
        print(f"{key:<22}{cat:<12}{basis}")
    print("\nCategories:", CATEGORIES)


if __name__ == "__main__":
    main()