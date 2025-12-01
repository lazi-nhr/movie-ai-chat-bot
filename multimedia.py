import pickle
import os
import numpy as np

def inspect_pkl(path, max_items=10):
    """
    Inspect the structure of a .pkl file.

    Parameters:
        path (str): path to the .pkl file
        max_items (int): how many dict/list entries to preview
    """

    print(f"=== Inspecting: {path} ===")

    if not os.path.exists(path):
        print("ERROR: File does not exist.")
        print()
        return

    with open(path, "rb") as f:
        obj = pickle.load(f)

    print("\n--- TYPE ---")
    print(type(obj))

    # If it's a numpy array
    if isinstance(obj, np.ndarray):
        print("\n--- NUMPY ARRAY SHAPE ---")
        print(obj.shape)
        print("\n--- DTYPE ---")
        print(obj.dtype)
        print("\n=== DONE ===\n")
        return

    # If it's a dictionary
    if isinstance(obj, dict):
        print("\n--- DICT SIZE ---")
        print(len(obj))

        keys = list(obj.keys())
        print("\n--- FIRST KEYS ---")
        for k in keys[:max_items]:
            print(" ", repr(k), " -> type:", type(obj[k]))

            # If value is an array
            if isinstance(obj[k], np.ndarray):
                print("      shape:", obj[k].shape, "dtype:", obj[k].dtype)

        print("\n=== DONE ===\n")
        return

    # If it's a list or tuple
    if isinstance(obj, (list, tuple)):
        print("\n--- LIST/TUPLE LENGTH ---")
        print(len(obj))

        print("\n--- FIRST ITEMS ---")
        for i, it in enumerate(obj[:max_items]):
            print(f"  index {i}: type={type(it)}")
            if isinstance(it, np.ndarray):
                print("      shape:", it.shape, "dtype:", it.dtype)

        print("\n=== DONE ===\n")
        return

    # Fallback for unknown/custom types
    print("\n--- VALUE (repr) ---")
    print(repr(obj))
    print("\n=== DONE ===\n")


if __name__ == "__main__":

    # Change these paths if needed
    paths = [
        "/space_mounts/atai-hs25/dataset/image_features/0000.pkl",
        "/space_mounts/atai-hs25/dataset/image_features/0347.pkl"
    ]

    for p in paths:
        inspect_pkl(p)