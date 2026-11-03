"""
P2 App — classify an uploaded big-cat photo with every model from Notebook 3, and explain each prediction.

Run from the repo root:

    streamlit run app/P2_Images-5_App_TeamName.py

What happens to an upload, in order:
  1. Decode and orient it the way a viewer sees it (phone photos store rotation as an EXIF flag).
  2. Apply EXACTLY the Notebook 2 preprocessing: RGB, centre square crop, Lanczos resize to 224x224, raw 0-255.
  3. Run it through all three joined models (Keras application + Notebook 3 probe), rebuilt exactly as in
     Notebooks 3 and 4. image_explainers applies each application's own preprocess_input itself, so the
     app hands it the raw 224x224 image and never preprocesses it a second time.
  4. For each model, one row: the prediction, a sorted probability bar plot, and raw saliency, SmoothGrad
     and Grad-CAM overlaid on the image.
"""
import gc
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')                       # draw figures off-screen; Streamlit displays them
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
MODELS_DIR = REPO_ROOT / 'models'
sys.path.insert(0, str(APP_DIR))            # use the copy of image_explainers.py that sits in app/

# TensorFlow's Intel oneDNN (MKL) CPU kernels can fail with "could not create a primitive" while computing
# the SmoothGrad gradients on a machine short of memory. Switching them off before TensorFlow is imported
# makes TensorFlow use its standard CPU kernels: slightly slower, same results.
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_DISABLE_MKL', '1')

import keras                              # noqa: E402  (after the path setup above)
from keras.applications import MobileNetV2, ResNet50, EfficientNetV2M   # noqa: E402
import image_explainers as xai              # noqa: E402

# The three Keras applications, smallest first, with the probe file Notebook 3 saved for each.
APPS = [
    ('mobilenetv2', 'MobileNetV2', MobileNetV2, '2.3 M parameters'),
    ('resnet50', 'ResNet50', ResNet50, '23.6 M parameters'),
    ('effnetv2m', 'EfficientNetV2M', EfficientNetV2M, '53.2 M parameters'),
]
TARGET_SIZE = 224        # Notebook 2, section 4.2
SMOOTH_SAMPLES = 32      # SmoothGrad noisy copies, as in Notebook 4


# ---------------------------------------------------------------------------------------------
# Loading: done once per server, then cached
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner='Loading the three models (first run downloads ImageNet weights)...')
def load_models():
    """Rebuild each Keras application exactly as in Notebook 3 and join it to its saved probe."""
    models = {}
    for key, _title, build, _size in APPS:
        keras_app = build(include_top=False, weights='imagenet', pooling='avg',
                          input_shape=(TARGET_SIZE, TARGET_SIZE, 3))
        probe = keras.models.load_model(MODELS_DIR / f'P2_N3_{key}.keras')
        models[key] = keras.Sequential([keras.Input((TARGET_SIZE, TARGET_SIZE, 3)), keras_app, probe])
    return models


@st.cache_data
def load_class_names():
    """Class names in label-index order, from the mapping Notebook 3 saved beside the probes."""
    label_map = json.loads((MODELS_DIR / 'label_map.json').read_text())
    return sorted(label_map, key=label_map.get)


# ---------------------------------------------------------------------------------------------
# Preprocessing: must match Notebook 2 exactly
# ---------------------------------------------------------------------------------------------
def preprocess(file_bytes):
    """
    Uploaded file -> (upright original, 224x224 uint8 RGB array), using Notebook 2's resize strategy:
    RGB, centre square crop of side min(width, height), Lanczos resize to 224x224, raw 0-255 pixels.
    Any mismatch with Notebook 2 here is the most common reason an app misclassifies images the
    notebooks got right.
    """
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)       # stand the photo upright before cropping
    img = img.convert('RGB')
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    square = img.crop((left, top, left + s, top + s)).resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    return img, np.asarray(square, dtype=np.uint8)


# ---------------------------------------------------------------------------------------------
# Inference and maps, cached per (image, model) so moving a slider does not recompute them
# ---------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def explain(file_bytes, key):
    _, img = preprocess(file_bytes)
    model = load_models()[key]
    probs = xai.predict_proba(model, img)
    pred = int(np.argmax(probs))
    np.random.seed(42)                        # SmoothGrad's noise, so the same upload gives the same map
    out = {'probs': probs, 'errors': {}}
    maps = {
        'raw': lambda: xai.saliency(model, img, class_index=pred),
        'smooth': lambda: xai.saliency(model, img, smooth=SMOOTH_SAMPLES, class_index=pred),
        'cam': lambda: xai.grad_cam(model, img, class_index=pred),
    }
    for name, compute in maps.items():
        # One map failing (usually memory, on a CPU) must not take down the whole page in a live demo.
        try:
            out[name] = compute()
        except Exception as error:
            out[name] = None
            out['errors'][name] = type(error).__name__
        gc.collect()
    return out


def probability_plot(probs, class_names, threshold):
    order = np.argsort(probs)[::-1]                    # most likely class first
    fig, ax = plt.subplots(figsize=(4, 3.4))
    colors = ['#2f7ed8' if k == order[0] else '#9aa5b1' for k in order]
    ax.bar([class_names[k] for k in order], probs[order], color=colors)
    ax.axhline(threshold, color='#e5484d', ls='--', lw=1, label=f'confidence threshold ({threshold:.2f})')
    ax.set_ylim(0, 1)
    ax.set_ylabel('probability')
    ax.legend(loc='upper right', fontsize=7, frameon=False)
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    return fig


def entropy_share(probs):
    """Prediction entropy as a share of its maximum: 0 = all on one class, 1 = spread evenly."""
    p = np.clip(probs, 1e-12, 1)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


# ---------------------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------------------
st.set_page_config(page_title='Spotted cat classifier', layout='wide')
st.title('Spotted cat classifier')
class_names = load_class_names()
st.markdown(
    f"Upload a photo of a **{', '.join(class_names[:-1])} or {class_names[-1]}**. Every model from Notebook 3 classifies it, "
    "smallest to largest, and shows **where it looked**: raw saliency, SmoothGrad saliency, and Grad-CAM."
)

with st.sidebar:
    st.header('Settings')
    threshold = st.slider(
        'Confidence threshold', min_value=0.0, max_value=1.0, value=0.70, step=0.05,
        help='Below this top-class probability, the app reports "not sure" instead of naming a class. '
             'The models can only answer with one of the five classes, so this is how the app admits that '
             'a photo may be none of them.')
    st.caption('The models know five classes only. A photo of anything else is still forced into one of them; '
               'the threshold and the spread of the probabilities are the only signs of that.')

upload = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png', 'webp', 'bmp'])
if upload is None:
    st.info('Upload an image to begin.')
    st.stop()

file_bytes = upload.getvalue()
try:
    original, img = preprocess(file_bytes)
except Exception as error:                      # not an image, or a corrupt one
    st.error(f'Could not read that file as an image ({type(error).__name__}: {error}).')
    st.stop()

left, right, _ = st.columns([2, 1, 1])
left.image(original, caption=f'Uploaded image ({original.width}×{original.height})')
right.image(img, caption='What the models see: centre crop, 224×224')

load_models()                                   # loads once, with its own spinner
summary = []
for key, title, _build, size in APPS:
    st.divider()
    with st.spinner(f'{title}: classifying and computing maps...'):
        out = explain(file_bytes, key)
    probs = out['probs']
    pred = int(np.argmax(probs))
    sure = probs[pred] >= threshold
    summary.append((title, class_names[pred], probs[pred], sure))

    st.subheader(f'{title}  ·  {size}')
    verdict = (f'**{class_names[pred]}** ({probs[pred]:.2f})' if sure
               else f'**not sure**: best guess {class_names[pred]} ({probs[pred]:.2f}), below the {threshold:.2f} threshold')
    st.markdown(f'Prediction: {verdict}  ·  spread of the probabilities: {entropy_share(probs):.2f} '
                f'(0 = all on one class, 1 = even across all five)')

    cols = st.columns(4)
    fig = probability_plot(probs, class_names, threshold)
    cols[0].pyplot(fig)
    plt.close(fig)
    for col, name, caption in [(cols[1], 'raw', 'saliency (raw)'),
                               (cols[2], 'smooth', f'SmoothGrad ({SMOOTH_SAMPLES} samples)'),
                               (cols[3], 'cam', 'Grad-CAM (7×7 grid, upsampled)')]:
        if out[name] is None:
            col.warning(f'{caption}: could not be computed on this machine ({out["errors"][name]}). '
                        'The prediction above is unaffected.')
        else:
            col.image(xai.overlay(img, out[name]), caption=caption)

st.divider()
st.subheader('Do the models agree?')
labels = {name for _, name, _, sure in summary if sure}
if len(labels) == 1 and all(sure for *_, sure in summary):
    st.success(f'All three models are confident it is **{labels.pop()}**.')
elif not labels:
    st.warning('No model is confident. The photo may not show any of the five classes, '
               'or the cat may be too small, hidden or blurred to identify.')
else:
    st.warning('The models disagree, or some are below the threshold: '
               + '; '.join(f'{t}: {n} ({p:.2f}{"" if s else ", not sure"})' for t, n, p, s in summary))
