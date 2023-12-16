import json
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mt.definitions import DATA_DIR


def extract_pylint_rating(pylint_output: str) -> float:
    # Regular expression to match the Pylint rating pattern
    match = re.search(r"Your code has been rated at ([0-9\.]+)/10", pylint_output)
    return float(match.group(1)) if match else 0


def process_commit_features(feats: dict[str, Any]) -> dict[str, int | float]:
    clean = {}
    for k, v in feats.items():
        if k == "loc":
            clean = clean | v[0]
        if k == "complexity":
            clean["complexity"] = v[0]
        if k == "dependencies":
            clean["dependencies"] = v[0]
        if k == "comment_tokens":
            clean["comment_tokens"] = v[0]
        if k == "quality":
            pylint_output = v[0] if v else ""
            clean["pylint_rating"] = extract_pylint_rating(pylint_output)
            # Count the number of Pylint messages (excluding the final rating line)
            pylint_messages = v[0].split("\n") if v else []
            clean["pylint_message_count"] = len(
                [
                    msg
                    for msg in pylint_messages
                    if msg.strip() and "Your code has been rated" not in msg
                ]
            )
    return clean


def aggregate_features(featurized_commits):
    aggregated_features = []

    for i, commit in enumerate(featurized_commits):
        commit_features = {}
        for file, features in commit.items():
            for key, value in features.items():
                if key in ["dependencies", "comment_tokens"]:
                    # Example: counting the number of items
                    value = len(value)
                # Aggregate features across files
                if key in commit_features:
                    commit_features[key].append(value)
                else:
                    commit_features[key] = [value]

        commit_features = process_commit_features(commit_features)
        # Create a new dictionary for aggregated features
        aggregated_commit_features = {
            k: np.mean(v) if v else 0 for k, v in commit_features.items()
        }
        aggregated_commit_features["commit_index"] = i  # Keeping track of commit index
        aggregated_features.append(aggregated_commit_features)

    return aggregated_features


with open(DATA_DIR / "dirty_features" / "feature_set.json") as f:
    featurized_commits = json.load(f)

# Load Y and Y_scale from JSON files
with open(DATA_DIR / "dirty_features" / "Y.json") as f:
    Y = json.load(f)

with open(DATA_DIR / "dirty_features" / "Y_scale.json") as f:
    Y_scale = json.load(f)

# Process the features
processed_features = aggregate_features(featurized_commits)

# Convert to DataFrame
df = pd.DataFrame(processed_features)

# Example: Standardizing the features (excluding commit_hash)
feature_cols = df.columns


# Ensure that Y, Y_scale, and processed_features are aligned
if len(Y) != len(Y_scale) or len(Y) != len(processed_features):
    raise ValueError("Mismatch in length of Y, Y_scale, and processed features")

# Add Y and Y_scale to DataFrame
df["Y"] = Y
df["Y_scale"] = Y_scale

# Create Y_scaled column
df["Y_scaled"] = df.apply(
    lambda row: row["Y"] / (row["Y_scale"] if row["Y_scale"] else 1), axis=1
)

Y_array = np.array(Y).reshape(-1, 1)

# Now df contains features, Y, Y_scale, and Y_scaled
df.to_csv(DATA_DIR / "final_non_scaled.csv")

# scaling
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
scaler = StandardScaler()
Y_standard_scaled = scaler.fit_transform(Y_array)
df["Y_standard_scaled"] = Y_standard_scaled.flatten()
df.to_csv("final.csv")
