import h5py

with h5py.File("/cs/labs/tsevi/lior.kotlar/amitai-s-thesis/from 2D to 3D/models 5.0/per wing/MODEL_18_POINTS_PER_WING_Jun 09 0.7-1.3 not reprojected/best_model.h5", "r") as f:
    keras_version = f.attrs.get("keras_version", "Unknown")
    backend = f.attrs.get("backend", "Unknown")
    tensorflow_version = f.attrs.get("tensorflow_version", "Unknown")

    print(f"Keras version: {keras_version}")
    print(f"Keras backend: {backend}")
    print(f"TensorFlow version: {tensorflow_version}")