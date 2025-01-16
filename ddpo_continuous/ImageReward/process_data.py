import os
import zipfile
import json
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image

base_dir = "data/imagereward"
for split in ["train", "test", "validation"]:
    os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)

for split in ["train", "test", "validation"]:
    source_dir = f"data/raw/ImageRewardDB/images/{split}"
    target_dir = f"data/imagereward/images/{split}"
    
    all_json_data = []
    
    # group by prompt_id, keeping the insertion order
    grouped_data = OrderedDict()
    
    print(f"Processing {split} dataset")
    # loop over all zip files
    # unzip and merge json files
    for zip_file in tqdm(sorted(os.listdir(source_dir))):
        if zip_file.endswith('.zip'):
            zip_path = os.path.join(source_dir, zip_file)
            folder_name = zip_file[:-4]
            extract_path = os.path.join(target_dir, folder_name)
            
            os.makedirs(extract_path, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
                
            json_file = os.path.join(extract_path, f"{folder_name}.json")
            if not os.path.exists(json_file):
                print(f"Warning: {json_file} does not exist")
                continue

            with open(json_file, 'r') as f:
                json_data = json.load(f)
            for item in json_data:
                prompt_id = item["prompt_id"]
                image_path = item["image_path"]

                if prompt_id not in grouped_data:
                    grouped_data[prompt_id] = {
                        "prompt_id": "",
                        "prompt": "",
                        "classification": "",
                        "image_amount_in_total": "",
                        "image_path": [],
                        "rank": [],
                        "overall_rating": [],
                        "image_text_alignment_rating": [],
                        "fidelity_rating": []
                    }
                
                group = grouped_data[prompt_id]
                
                # set fixed fields
                if not group["prompt_id"]:
                    group["prompt_id"] = prompt_id
                    group["prompt"] = item["prompt"]
                    group["classification"] = item["classification"]
                    group["image_amount_in_total"] = item["image_amount_in_total"]
                else:
                    assert group["prompt"] == item["prompt"]
                    assert group["classification"] == item["classification"]
                    assert group["image_amount_in_total"] == item["image_amount_in_total"]
                # append list fields
                group["image_path"].append(item["image_path"])
                group["rank"].append(item["rank"])
                group["overall_rating"].append(item["overall_rating"])
                group["image_text_alignment_rating"].append(item["image_text_alignment_rating"])
                group["fidelity_rating"].append(item["fidelity_rating"])
    
    # convert grouped data to list
    reorganized_data = list(grouped_data.values())

    # save reorganized json
    merged_json_path = os.path.join(base_dir, f"{split}.json")
    with open(merged_json_path, 'w') as f:
        json.dump(reorganized_data, f)