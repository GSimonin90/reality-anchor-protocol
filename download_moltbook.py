import pandas as pd
from datasets import load_dataset
import random

print("Downloading dataset 'SimulaMet/moltbook-observatory-archive'...")

# 1. DOWNLOAD
try:
    dataset = load_dataset("SimulaMet/moltbook-observatory-archive", "posts", split="archive")
    print(f"Dataset downloaded. Found {len(dataset)} interactions.")
except Exception as e:
    print(f"Error downloading: {e}")
    exit()

# 2. CONVERT TO PANDAS
df_raw = dataset.to_pandas()

# 3. MAPPING (FIXED: Using 'agent_name')
dashboard_data = []
print("Processing data using column 'agent_name'...")

for index, row in df_raw.iterrows():
    try:
        # Get Content
        content = row.get('content', '') or row.get('title', '') # Sometimes content is in title
        if not isinstance(content, str) or len(content) < 5:
            continue

        # Get Agent Name (FIXED HERE)
        # We prefer 'agent_name' (readable), fallback to 'agent_id'
        agent_raw = row.get('agent_name')
        if not agent_raw:
             agent_raw = row.get('agent_id', 'Unknown')
             
        agent_id = str(agent_raw)

        # Belief Score (Simulated for visualization)
        # In the real Moltbook, 'score' is usually just upvotes, not truth.
        # So we keep the random simulation for the color gradient.
        belief = random.uniform(0.1, 0.9)

        dashboard_data.append({
            "timestamp": index,
            "real_timestamp": row.get('created_at', ''),
            "agent_id": f"@{agent_id}", # Add @ for style
            "content": content,
            "belief_score": round(belief, 4)
        })
    except Exception as e:
        continue

# 4. SAVE
df_final = pd.DataFrame(dashboard_data)

# Save last 5000 rows
if len(df_final) > 5000:
    df_final = df_final.tail(5000)

filename = "moltbook_REAL_data.csv"
df_final.to_csv(filename, index=False)

print(f"Success! Created '{filename}' with {len(df_final)} rows.")
print(f"Unique agents found: {df_final['agent_id'].nunique()}")
print("Now upload this file to the Dashboard!")