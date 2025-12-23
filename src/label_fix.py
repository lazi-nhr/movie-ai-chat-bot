import json
import requests
import re
from tqdm import tqdm
import time
import os

def is_qid(label):
    """Check if the label is a QID (e.g., Q95873202)"""
    return bool(re.match(r'^Q\d+$', label))

def get_wikidata_labels_batch(qids):
    """Fetch labels for multiple QIDs in one request using Wikidata API"""
    if not qids:
        return {}
    
    # Wikidata allows up to 50 entities per request
    batch_size = 1
    all_labels = {}
    missing_count = 0
    no_english_label_count = 0
    
    # Set up headers with User-Agent (required by Wikidata)
    headers = {
        'User-Agent': 'LabelFixBot/1.0 (Python/requests)'
    }
    
    # Calculate number of batches for progress bar
    num_batches = (len(qids) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(qids), batch_size), total=num_batches, desc="Fetching batches"):
        batch = qids[i:i+batch_size]
        ids = "|".join(batch)
        
        url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbgetentities',
            'ids': ids,
            'props': 'labels',
            'languages': 'en',
            'format': 'json'
        }
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Debug: print first batch response structure
                if i == 0:
                    tqdm.write("\n=== FIRST BATCH DEBUG ===")
                    tqdm.write(f"Response keys: {list(data.keys())}")
                    if 'entities' in data:
                        tqdm.write(f"Number of entities returned: {len(data['entities'])}")
                        sample_qid = list(data['entities'].keys())[0] if data['entities'] else None
                        if sample_qid:
                            tqdm.write(f"Sample QID: {sample_qid}")
                            sample_entity = data['entities'][sample_qid]
                            tqdm.write(f"Sample entity keys: {list(sample_entity.keys())}")
                            if 'labels' in sample_entity:
                                tqdm.write(f"Sample labels: {sample_entity['labels']}")
                    tqdm.write("=== END DEBUG ===\n")
                
                if 'entities' in data:
                    for qid, entity in data['entities'].items():
                        # Check if entity exists (not missing)
                        if 'missing' in entity:
                            missing_count += 1
                            if missing_count <= 3:
                                tqdm.write(f"Missing entity: {qid}")
                            continue
                        
                        if 'labels' in entity:
                            if 'en' in entity['labels']:
                                all_labels[qid] = entity['labels']['en']['value']
                            elif entity['labels']:
                                # Get first available label if English not available
                                first_lang = list(entity['labels'].keys())[0]
                                all_labels[qid] = entity['labels'][first_lang]['value']
                                no_english_label_count += 1
                                if no_english_label_count <= 3:
                                    tqdm.write(f"No English label for {qid}, using {first_lang}: {entity['labels'][first_lang]['value']}")
                        else:
                            if missing_count + no_english_label_count <= 5:
                                tqdm.write(f"No labels at all for {qid}")
                                
            else:
                tqdm.write(f"HTTP {response.status_code} for batch starting at index {i}")
                if i == 0:
                    tqdm.write(f"Response text: {response.text[:500]}")
                
            time.sleep(0.1)  # Small delay between batch requests
        except Exception as e:
            tqdm.write(f"Error fetching batch at index {i}: {e}")
    
    tqdm.write("\n=== SUMMARY ===")
    tqdm.write(f"Total labels fetched: {len(all_labels)}")
    tqdm.write(f"Missing entities: {missing_count}")
    tqdm.write(f"Entities without English labels: {no_english_label_count}")
    
    return all_labels

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(base_dir, "cache/entities/identifier_to_label.json")
    
    # Load the JSON file
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    
    # Find all QIDs that need labels
    qids_to_fetch = []
    url_to_qid = {}
    
    for url, label in data.items():
        if is_qid(label):
            qids_to_fetch.append(label)
            url_to_qid[url] = label
    
    print(f"QIDs needing labels: {len(qids_to_fetch)}")
    print(f"Sample QIDs: {qids_to_fetch[:5]}")
    
    if not qids_to_fetch:
        print("No labels to fetch!")
        return
    
    # Fetch all labels in batches
    qid_to_label = get_wikidata_labels_batch(qids_to_fetch)
    
    print(f"\nSuccessfully fetched {len(qid_to_label)} labels")
    print(f"Failed to fetch: {len(qids_to_fetch) - len(qid_to_label)} labels")
    
    # Show some QIDs that failed
    failed_qids = [qid for qid in qids_to_fetch if qid not in qid_to_label]
    if failed_qids:
        print(f"Sample failed QIDs: {failed_qids[:10]}")
    
    # Update the data
    updates_made = 0
    for url, qid in tqdm(url_to_qid.items(), desc="Updating entries"):
        if qid in qid_to_label:
            new_label = qid_to_label[qid]
            if new_label != qid:
                data[url] = new_label
                updates_made += 1
                if updates_made <= 10:  # Show first 10 updates
                    tqdm.write(f"Updated {qid} -> {new_label}")
    
    # Save all changes at once
    print("\nSaving changes...")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nCompleted! Total updates made: {updates_made}")

if __name__ == "__main__":
    main()