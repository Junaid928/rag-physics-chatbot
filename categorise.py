import json

# Define which arXiv categories count as "physics"
physics_prefixes = [
    "physics", "astro-ph", "cond-mat", "gr-qc", "hep-ex",
    "hep-lat", "hep-ph", "hep-th", "nucl-ex", "nucl-th", "quant-ph"
]

def is_physics_category(categories):
    return any(cat.startswith(tuple(physics_prefixes)) for cat in categories.split())

# Paths
input_path = r"C:\Users\jnk47\Documents\datasets\arxiv-metadata-oai-snapshot.json"
output_path = r"C:\Users\jnk47\Documents\datasets\arxiv_physics_subset.json"

# Set how many entries to extract (adjust as needed)
MAX_ENTRIES = 10000

physics_entries = []
with open(input_path, 'r') as infile:
    for line in infile:
        entry = json.loads(line)
        if is_physics_category(entry.get("categories", "")):
            physics_entries.append({
                "id": entry["id"],
                "title": entry["title"].strip(),
                "abstract": entry["abstract"].strip(),
                "categories": entry["categories"]
            })
        if len(physics_entries) >= MAX_ENTRIES:
            break

# Save filtered entries to a smaller JSON file
with open(output_path, 'w') as outfile:
    json.dump(physics_entries, outfile, indent=2)

print(f"Saved {len(physics_entries)} physics-related entries to {output_path}")
