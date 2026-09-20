"""Fix the Path.exists bug in notebook 02 and verify."""
import json
from pathlib import Path

nb_path = Path("notebooks/02_data_preprocessing.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

OLD = 'missing_files = ~metadata_df["image_path"].map(Path.exists)'
NEW = 'missing_files = ~metadata_df["image_path"].map(lambda p: Path(p).exists())'

fixed = False
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell["source"])
    if OLD in src:
        cell["source"] = [src.replace(OLD, NEW)]
        print(f"Fixed cell {i}")
        fixed = True
    if NEW in src and not fixed:
        print(f"Cell {i} already has the fix.")

if not fixed and not any(NEW in "".join(c.get("source","")) for c in nb["cells"]):
    print("ERROR: Pattern not found in any cell!")
    # Print all code cells for debugging
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "missing_files" in src:
                print(f"Cell {i} has 'missing_files':")
                print(src)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify
with open(nb_path, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

print("\n--- Verification ---")
for i, cell in enumerate(nb2["cells"]):
    src = "".join(cell.get("source", []))
    if "missing_files" in src:
        for line in src.splitlines():
            if "missing_files" in line and "Path" in line:
                print(f"Cell {i}: {line}")
