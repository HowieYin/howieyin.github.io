# Reproduce on CPU

Python 3.12.2; one thread; dependencies pinned in requirements.txt.
Real scikit-learn digits dataset; values /16; stratified split seed 42.
The 1797 images are a copy of the UCI test portion; this educational experiment creates its own 1347/450 split.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiment.py --out results
```

Train for the fixed number of epochs; do not select a checkpoint using the test set.
Dataset hash and exact split indices are in results.json. Small numerical differences may occur with different backends.
Saved state_dict files: model.pt.
Load with torch.load(path, map_location="cpu", weights_only=True) and the model definition in experiment.py.

This is a floating-point CPU teaching experiment. It makes no GPU/NPU speed claims.
For FGSM, epsilon is defined on raw/16 pixels, attacks all 450 inputs, and reports both full-set accuracy and clean-correct conditional success.
For BN, only folded convolution weights are fake-quantized in the optional INT8 check; activations and biases remain float.
