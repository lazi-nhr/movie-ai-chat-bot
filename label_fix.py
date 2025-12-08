import json
import requests
import re
from tqdm import tqdm
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def is_qid(label):
    """Check if the label is a QID (e.g., Q95873202)"""
    return bool(re.match(r'^Q\d+$', label))

def get_wikidata_labels_batch(qids):
    """Fetch labels for multiple QIDs in one request using Wikidata API"""
    if not qids:
        return {}
    
    # Wikidata allows up to 50 entities per request
    batch_size = 50
    all_labels = {}
    
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
                
                # Debug: print first batch response
                if i == 0:
                    tqdm.write(f"Sample response keys: {list(data.keys())}")
                    if 'entities' in data:
                        sample_qid = list(data['entities'].keys())[0] if data['entities'] else None
                        if sample_qid:
                            tqdm.write(f"Sample entity: {sample_qid}")
                            tqdm.write(f"Sample entity data: {data['entities'][sample_qid]}")
                
                if 'entities' in data:
                    for qid, entity in data['entities'].items():
                        # Check if entity exists (not missing)
                        if 'missing' not in entity:
                            if 'labels' in entity and 'en' in entity['labels']:
                                all_labels[qid] = entity['labels']['en']['value']
                            elif 'labels' in entity and entity['labels']:
                                # Get first available label if English not available
                                first_lang = list(entity['labels'].keys())[0]
                                all_labels[qid] = entity['labels'][first_lang]['value']
            else:
                tqdm.write(f"HTTP {response.status_code} for batch starting at index {i}")
                if i == 0:
                    tqdm.write(f"Response text: {response.text[:500]}")
                
            time.sleep(0.1)  # Small delay between batch requests
        except Exception as e:
            tqdm.write(f"Error fetching batch at index {i}: {e}")
    
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
    
    print(f"Successfully fetched {len(qid_to_label)} labels")
    
    # Update the data
    updates_made = 0
    for url, qid in tqdm(url_to_qid.items(), desc="Updating entries"):
        if qid in qid_to_label:
            new_label = qid_to_label[qid]
            if new_label != qid:
                data[url] = new_label
                updates_made += 1
                if updates_made <= 5:  # Show first 5 updates
                    tqdm.write(f"Updated {qid} -> {new_label}")
    
    # Save all changes at once
    print("Saving changes...")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nCompleted! Total updates made: {updates_made}")

if __name__ == "__main__":
    main()