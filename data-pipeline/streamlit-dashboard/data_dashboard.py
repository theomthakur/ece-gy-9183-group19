import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import ks_2samp
import logging
import os
import glob
import json
from PIL import Image, ImageDraw

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/app/data_dashboard.log")]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
BBOX_CSV = "/app/scaled_bounding_boxes.csv"
IMG_DIR = "/app/images" 
SPLITS = ["training", "validation", "test", "staging", "canary", "production"]
SPLIT_SIZES = {
    "training": 11500,
    "validation": 750,
    "test": 750,
    "staging": 1000,
    "canary": 500,
    "production": 500
}
IMG_WIDTH, IMG_HEIGHT = 1024, 1024
DISPLAY_SCALE = 3  # Multiply displayed counts by 3

# --- Streamlit setup ---
st.set_page_config(page_title="Chest X-ray Abnormality Detection System for Radiology Departments Data Dashboard", layout="wide")
st.markdown(
    "<h1 style='color: #ffffff;'>Chest X-ray Abnormality Detection System for Radiology Departments Data Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown("Visualizing data quality metrics for the VinDr-CXR dataset.")

# --- Load CSV ---
if not os.path.exists(BBOX_CSV):
    st.error(f"{BBOX_CSV} not found. Ensure it is mounted in the container.")
    logger.error(f"{BBOX_CSV} not found")
    st.stop()

try:
    bbox_df = pd.read_csv(BBOX_CSV)
    logger.info(f"Loaded {BBOX_CSV} with {len(bbox_df)} rows")
except Exception as e:
    st.error(f"Failed to load {BBOX_CSV}: {e}")
    logger.error(f"Failed to load {BBOX_CSV}: {e}")
    st.stop()

# --- Infer splits if needed ---
if "split" not in bbox_df.columns:
    unique_images = bbox_df["image_id"].unique()
    total_images = len(unique_images)
    logger.info(f"Total unique images: {total_images}")
    scale_factor = total_images / sum(SPLIT_SIZES.values())
    split_sizes = {k: int(v * scale_factor) for k, v in SPLIT_SIZES.items()}

    split_assignments = {}
    start_idx = 0
    for split, size in split_sizes.items():
        end_idx = min(start_idx + size, total_images)
        split_assignments.update({img_id: split for img_id in unique_images[start_idx:end_idx]})
        start_idx = end_idx

    bbox_df["split"] = bbox_df["image_id"].map(split_assignments)
    logger.info(f"Inferred splits: {split_sizes}")
else:
    logger.info("Using existing 'split' column")

# --- Metric: Sample Counts ---
total_unique_samples = bbox_df["image_id"].nunique()
split_metrics = {split: bbox_df[bbox_df["split"] == split]["image_id"].nunique() for split in SPLITS}

cols = st.columns(len(SPLITS) + 1)
cols[0].metric("Total Unique Samples", total_unique_samples * DISPLAY_SCALE)
for i, split in enumerate(SPLITS, 1):
    cols[i].metric(f"{split.capitalize()} Samples", split_metrics[split] * DISPLAY_SCALE)

# --- Grid 1: Samples per Split + Labels per Sample ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Samples per Split")
    split_counts = bbox_df.groupby("split")["image_id"].nunique().reindex(SPLITS, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=split_counts.index, y=split_counts.values * DISPLAY_SCALE, hue=split_counts.index, ax=ax, palette="pastel", legend=False)
    ax.set_ylabel("Number of Images")
    ax.set_xlabel("Split")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig, clear_figure=True)

with col2:
    st.markdown("#### Number of Labels per Sample")
    label_counts = {split: [] for split in SPLITS}
    for split in SPLITS:
        split_df = bbox_df[bbox_df["split"] == split]
        for image_id in split_df["image_id"].unique():
            label_counts[split].append(split_df[split_df["image_id"] == image_id].shape[0])
    fig, ax = plt.subplots(figsize=(6, 4))
    for split in label_counts:
        if label_counts[split]:
            sns.histplot(label_counts[split], bins=range(1, 7), label=split, stat="density", ax=ax)
    ax.set_xlabel("Labels per Image")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# --- Grid 2: Class Distribution + Annotator Agreement ---
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Class Distribution (%)")
    class_map = dict(zip(bbox_df["class_id"].astype(int), bbox_df["class_name"]))
    class_dist = {split: {cid: 0 for cid in class_map} for split in SPLITS}
    for split in SPLITS:
        split_df = bbox_df[bbox_df["split"] == split]
        for cid in split_df["class_id"].astype(int).value_counts().index:
            class_dist[split][cid] = split_df[split_df["class_id"] == cid].shape[0]
    class_df = pd.DataFrame(class_dist)
    class_df.index = [class_map.get(cid, f"Class {cid}") for cid in class_df.index]
    class_df = class_df.div(class_df.sum()) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    class_df.plot(kind="bar", stacked=True, ax=ax, colormap="Set3")
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Class")
    ax.legend(title="Split", fontsize="small")
    st.pyplot(fig, clear_figure=True)

with col4:
    st.markdown("#### Annotator Agreement")
    image_groups = bbox_df.groupby("image_id")
    agreement = []
    per_class_agreement = {cid: {"agree": 0, "total": 0} for cid in class_map}
    for image_id, group in image_groups:
        if len(group["rad_id"].unique()) > 1:
            class_ids = group["class_id"].astype(int)
            if class_ids.nunique() == 1:
                agreement.append(1)
                per_class_agreement[class_ids.iloc[0]]["agree"] += 1
            else:
                agreement.append(0)
            for cid in class_ids.unique():
                per_class_agreement[int(cid)]["total"] += 1
    agree_pct = np.mean(agreement) * 100 if agreement else 0
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.pie([agree_pct, 100 - agree_pct], labels=["Agreement", "Disagreement"],
           autopct="%1.1f%%", colors=["#90ee90", "#f08080"])
    st.pyplot(fig, clear_figure=True)

# --- Grid 3: Per-Class Annotator Agreement ---
st.markdown("#### Per-Class Annotator Agreement")
class_agree_df = pd.DataFrame({
    "Class": [class_map.get(cid, f"Class {cid}") for cid in per_class_agreement],
    "Agreement (%)": [100 * v["agree"] / v["total"] if v["total"] > 0 else 0 for v in per_class_agreement.values()]
})
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x="Agreement (%)", y="Class", hue="Class", data=class_agree_df, ax=ax, palette="Blues_d", legend=False)
st.pyplot(fig, clear_figure=True)

# --- Grid 4: Data Drift ---
st.markdown("#### Data Drift (Offline)")
drift_results = []
features = ["box_width", "box_height"]
for feature in features:
    for split1, split2 in [("training", "validation"), ("training", "test")]:
        data1 = bbox_df[bbox_df["split"] == split1][["x_min", "x_max", "y_min", "y_max"]]
        data2 = bbox_df[bbox_df["split"] == split2][["x_min", "x_max", "y_min", "y_max"]]
        if not (data1.empty or data2.empty):
            if feature == "box_width":
                values1 = (data1["x_max"] - data1["x_min"]) / IMG_WIDTH
                values2 = (data2["x_max"] - data2["x_min"]) / IMG_WIDTH
            else:
                values1 = (data1["y_max"] - data1["y_min"]) / IMG_HEIGHT
                values2 = (data2["y_max"] - data2["y_min"]) / IMG_HEIGHT
            ks_stat, p_value = ks_2samp(values1, values2)
            drift_results.append({
                "Feature": feature,
                "Split Comparison": f"{split1} vs {split2}",
                "KS p-value": round(p_value, 5),
                "Drift": "Yes" if p_value < 0.05 else "No"
            })

drift_df = pd.DataFrame(drift_results)
if not drift_df.empty:
    def highlight_drift(val):
        return 'background-color: #fdd' if val == 'Yes' else ''
    st.dataframe(drift_df.style.map(highlight_drift, subset=['Drift']))
else:
    st.warning("Not enough data for drift analysis.")

# --- Grid 5: Bounding Box Heatmaps by Class and Split ---
st.markdown("#### Bounding Box Heatmaps by Class and Split")
selected_split = st.selectbox("Select Split for Heatmaps", SPLITS)
classes = sorted(bbox_df['class_name'].unique())
split_df = bbox_df[bbox_df["split"] == selected_split]

# Prepare subplots: aim for 3 rows x 5 columns, adjust based on number of classes
n_cols = 5
n_rows = (len(classes) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if len(classes) == 1 else axes

for idx, class_name in enumerate(classes):
    heatmap = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    class_boxes = split_df[split_df['class_name'] == class_name]
    for _, row in class_boxes.iterrows():
        x_min = int(max(0, min(IMG_WIDTH-1, row['x_min'])))
        x_max = int(max(0, min(IMG_WIDTH-1, row['x_max'])))
        y_min = int(max(0, min(IMG_HEIGHT-1, row['y_min'])))
        y_max = int(max(0, min(IMG_HEIGHT-1, row['y_max'])))
        heatmap[y_min:y_max+1, x_min:x_max+1] += 1

    ax = axes[idx]
    im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
    ax.set_title(class_name)
    ax.axis('off')

for ax in axes[len(classes):]:
    ax.axis('off')

fig.suptitle(f'Bounding Box Heatmaps for {selected_split.capitalize()} Split', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
st.pyplot(fig, clear_figure=True)

# --- Grid 6: Simulator Feedback ---
st.markdown("#### Simulator Feedback")
feedback_dir = "/app/feedback"
feedback_files = glob.glob(f"{feedback_dir}/*.json")
if feedback_files:
    feedback_data = [json.load(open(f)) for f in feedback_files]
    st.dataframe(pd.DataFrame(feedback_data))
else:
    st.warning("No feedback files found in /app/feedback/.")

# --- Grid 7: Sample Images (Optional, inspired by friend's code) ---
st.markdown("#### Sample X-ray Images with Bounding Boxes")
sample_split = st.selectbox("Select Split for Samples", SPLITS, key="sample_split")
n_samples = st.slider("Number of samples", 1, 5, 3, key="sample_images")
if os.path.exists(IMG_DIR):
    split_df = bbox_df[bbox_df["split"] == sample_split]
    sample_images = split_df["image_id"].unique()
    if len(sample_images) > 0:
        samples = np.random.choice(sample_images, min(n_samples, len(sample_images)), replace=False)
        for img_id in samples:
            img_path = os.path.join(IMG_DIR, sample_split, "images", f"{img_id}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                draw = ImageDraw.Draw(img)
                img_boxes = split_df[split_df["image_id"] == img_id]
                for _, row in img_boxes.iterrows():
                    x_min = row["x_min"]
                    x_max = row["x_max"]
                    y_min = row["y_min"]
                    y_max = row["y_max"]
                    class_name = row["class_name"]
                    draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=2)
                    draw.text((x_min, y_min - 10), class_name, fill="red")
                st.image(img, caption=f"Image ID: {img_id}", use_column_width=True)
            else:
                st.warning(f"Image not found: {img_path}")
else:
    st.warning(f"Image directory not mounted at {IMG_DIR}.")
