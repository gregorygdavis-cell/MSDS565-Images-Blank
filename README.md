# MSDS565-Images-Blank

**Telling five spotted wild cats apart from scraped web photos:** `cheetah`, `jaguar`, `leopard`, `ocelot` and `serval`.

The project follows the full CRISP-DM pipeline on a dataset built from scratch. It scrapes and cleans about 2,000 web images, compares
three pretrained Keras applications of very different sizes using linear probes, explains what the models look at with saliency
and Grad-CAM, and deploys all three in a Streamlit app.

## Classes and why

All five are tan-to-golden cats covered in black markings, and are easy to confuse. **Coat pattern, build and face** separate them:

| Class | Coat pattern | Build | Face / head |
|---|---|---|---|
| `cheetah` | solid round spots | large, slender, long-legged | black "tear marks" from eye to mouth |
| `serval` | solid spots merging into bars on the neck | medium, extremely long legs | small head, huge upright ears |
| `leopard` | small, open rosettes, tightly packed | large, long body | no tear marks |
| `jaguar` | larger rosettes with a dot inside | large, stocky, heavy-set | very broad head |
| `ocelot` | elongated, chain-like blotches in lines | small and compact | two black stripes on each cheek |

The classes come in look-alike pairs: **leopard ↔ jaguar** (rosettes), **serval ↔ cheetah** (solid spots, long legs) and
**ocelot ↔ jaguar/leopard** (open-centred markings). In a photo there is rarely anything to show scale, so size does not help.
Leopard, jaguar and cheetah are ImageNet classes. Ocelot and serval are deliberately **not**, which tests whether the pretrained
embeddings capture coat pattern in general. Before any modelling, Notebook 1 predicted that **leopard ↔ jaguar** would be the
most-confused pair.

## How the dataset was built

Images were scraped from Bing image search with the provided `ddgs_scraper.py` (imported unmodified):
400 images per class, 200 search results per query, a minimum of 224×224 pixels, aspect ratio 0.5–2.0, a 10 s timeout per image,
the `bing` backend, and the query-match check on. Every argument is justified in Notebook 1.

**Search terms**, five per class. They follow the same pattern for every class, and the zoo slice gives all five a shared background:

| Slice | leopard | jaguar | cheetah | ocelot | serval |
|---|---|---|---|---|---|
| General | `leopard wildlife` | `jaguar big cat wildlife` | `cheetah wildlife` | `ocelot wild cat` | `serval wild cat` |
| Behaviour | `leopard resting in tree` | `jaguar swimming river` | `cheetah running` | `ocelot on tree branch` | `serval cat pouncing` |
| Habitat | `african leopard savanna` | `jaguar animal Pantanal` | `cheetah savanna` | `ocelot rainforest` | `serval savanna grassland` |
| Close-up | `leopard face close up wildlife` | `jaguar animal face close up` | `cheetah face close up wildlife` | `ocelot face close up animal` | `serval face close up animal` |
| Zoo | `leopard in zoo` | `jaguar animal in zoo` | `cheetah in zoo` | `ocelot animal in zoo` | `serval animal in zoo` |

"jaguar", "ocelot" and "serval" each have other meanings (a car brand, a video-game character, an anime character), so their queries
always include an animal word.

**From 2,000 scraped images to 1,771 kept:**

| Step | Removed | Notes |
|---|---|---|
| Scrape | — | every class reached 400 downloaded and 400 readable, on the first attempt |
| Hand cleaning | 45 | rejects were moved into reason folders and counted from disk: 29 other species, 11 drawings or anime, 5 unusable |
| Duplicates | 184 | 64-bit difference hashes across all classes, threshold 9 of 64 bits, chosen from a figure of sample pairs at each distance. Every copy of a cross-class duplicate was removed |
| **Kept** | **1,771** | 348–374 per class (imbalance ratio 1.07). Split 80/20, stratified, `random_state=42`: 1,416 train / 355 test |

Each image is converted to RGB and **centre-cropped to a square, then resized to 224×224** with Lanczos resampling. Padding was rejected
because image shape differs by class, so the bars would carry class information. Stretching was rejected because it distorts body build.

**No images are in this repo.** They are copyrighted by their creators and are used under fair use for coursework only.
`data/P2_manifest.csv` is the record of the dataset: one row per scraped image, with its `filename`, `label`, `query`,
`source_url`, `image_url`, `readable`, `kept`, `removal_reason` and `split`.

## Notebooks, in order

| Notebook | What it does | What it produces |
|---|---|---|
| `notebooks/P2_Images-1_Scrape_Blank.ipynb` | Declares and justifies the classes and queries, scrapes with automatic retries when throttled, checks every download opens | images in the local data folder; `data/P2_manifest.csv` |
| `notebooks/P2_Images-2_Preprocessing_Blank.ipynb` | Hand-cleaning summary, duplicate removal, image EDA (sizes, mean images, grayscale share, class support), resize decision, split | `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `files_*.csv`, `label_map.json`, `preprocess.json` in the local data folder; `kept`, `removal_reason`, `split` columns in the manifest |
| `notebooks/P2_Images-3_TransferLearning_Blank.ipynb` | Balance (oversample), then augment, then embed with MobileNetV2, ResNet50 and EfficientNetV2M; one linear probe each; evaluation and comparison | `models/P2_N3_mobilenetv2.keras`, `models/P2_N3_resnet50.keras`, `models/P2_N3_effnetv2m.keras`, `models/label_map.json` |
| `notebooks/P2_Images-4_xAI_Blank.ipynb` | Saliency, SmoothGrad and Grad-CAM with the provided `image_explainers.py`; shortcut hunt with average Grad-CAM and occlusion ablations | the findings below |

## Results

| Keras application | Params (no top) | Test accuracy | Macro-F1 | Embedding time (CPU) |
|---|---|---|---|---|
| MobileNetV2 | 2.3 M | 94.4% | 0.944 | 95 s |
| ResNet50 | 23.6 M | 94.9% | 0.950 | 214 s |
| EfficientNetV2M | 53.2 M | 94.9% | 0.949 | 482 s |

- **A 23-fold larger model buys nothing measurable.** The gap between the smallest and the largest model is 2 test images out of 355
- **The Notebook 1 prediction held:** leopard ↔ jaguar is the most-confused pair for all three models. Surprisingly, `ocelot`, a non-ImageNet
  class, has the *highest* recall
- **Shortcut hunt:** stock-photo watermarks and banners are **not** used (greying the banner band costs no more than greying the top band).
  Background is **partly** used (the edges of the image alone give 57–81% accuracy). One test "error" is actually a mislabelled leopard
  that the model got right

## Running it yourself

1. **Environment:** `conda env create -f environment.yml`, then `conda activate msds565`
2. **Folders:** the notebooks expect this repo at `~/P2_Images` and keep all images and arrays in `~/MSDS565_data`, outside the repo.
   To use other locations, edit `PROJECT_DIR` (and `IMAGE_DIR` / `PREP_DIR`) in the first code cell of each notebook
3. **Order:** run Notebooks 1 → 4. Notebook 1 saves each class's scrape records, so re-running it reloads them rather than scraping again.
   Search results change over time, so a fresh scrape would not reproduce this dataset. The manifest is the record of what was used.
   Re-running Notebook 1 rewrites the manifest, so re-run Notebook 2 afterwards to restore its columns

## Launching the app

From the repo root:

```
conda activate msds565
streamlit run app/P2_Images-5_App_Blank.py
```

Upload a photo, and all three models classify it, one row per model, smallest first. Each row shows a sorted probability bar plot and
raw saliency, SmoothGrad and Grad-CAM maps. The upload is preprocessed exactly as in Notebook 2 (centre crop, 224×224). A **confidence threshold**
in the sidebar (default 0.70) makes a model report "not sure" when its top probability is below the threshold, because the models can only
answer with one of the five classes.

The first run downloads the ImageNet weights (about 320 MB). On a CPU, the first upload of each image takes 1–3 minutes, mostly
EfficientNetV2M's SmoothGrad. Results are cached while the app runs, so repeat uploads are instant.

## Model files

Each saved probe is a single Dense layer, 90–136 KB, so all three are committed directly and **Git LFS is not needed**. The Keras
applications themselves are not stored: Notebook 4 and the app rebuild them from `keras.applications` with ImageNet weights and the
same arguments as Notebook 3 (`include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3)`).

## Repo layout

```
MSDS565-Images-Blank/
├── README.md
├── environment.yml
├── data/
│   └── P2_manifest.csv
├── models/
│   ├── P2_N3_mobilenetv2.keras
│   ├── P2_N3_resnet50.keras
│   ├── P2_N3_effnetv2m.keras
│   └── label_map.json
├── notebooks/
│   ├── ddgs_scraper.py            # provided, unmodified
│   ├── image_explainers.py        # provided, unmodified
│   ├── P2_Images-1_Scrape_Blank.ipynb
│   ├── P2_Images-2_Preprocessing_Blank.ipynb
│   ├── P2_Images-3_TransferLearning_Blank.ipynb
│   └── P2_Images-4_xAI_Blank.ipynb
└── app/
    ├── image_explainers.py        # provided, second copy for the app
    └── P2_Images-5_App_Blank.py
```
