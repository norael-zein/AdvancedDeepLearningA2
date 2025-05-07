import json
import os

# === Config ===
dataset_dir = "dataset/"
splits = ["train", "valid", "test"]

# === Step 1: Extract all used category_ids ===
used_category_ids = set()
category_id_to_name = {}

for split in splits:
    with open(f"dataset/{split}/_annotations.coco.json") as f:
        data = json.load(f)
        for ann in data["annotations"]:
            used_category_ids.add(ann["category_id"])
        for cat in data["categories"]:
            if cat["id"] in used_category_ids:
                category_id_to_name[cat["id"]] = cat["name"]

# === Step 2: Build a unified mapping, re-index from 0 ===
old_to_new_id = {}
new_categories = []
for new_id, (old_id, name) in enumerate(sorted(category_id_to_name.items())):
    old_to_new_id[old_id] = new_id
    new_categories.append({"id": new_id, "name": name})

# === Step 3: Clean each file ===
for split in splits:
    input_path = os.path.join(dataset_dir, split, "_annotations.coco.json")
    output_path = os.path.join(dataset_dir, split, "_annotations_cleaned.coco.json")

    with open(input_path) as f:
        coco = json.load(f)

    # Replace category_id in annotations
    new_annotations = []
    for ann in coco["annotations"]:
        if ann["category_id"] in old_to_new_id:
            ann["category_id"] = old_to_new_id[ann["category_id"]]
            new_annotations.append(ann)

    coco["annotations"] = new_annotations
    coco["categories"] = new_categories

    with open(output_path, "w") as f:
        json.dump(coco, f)

    print(f"✅ {split} set cleaned → {output_path}")

# === Output new category list ===
print("\n🧾 Cleaned categories:")
for cat in new_categories:
    print(f"  [{cat['id']}] {cat['name']}")