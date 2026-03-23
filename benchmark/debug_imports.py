"""Debug: check directory layout and model imports."""
import sys
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
print(f"PROJECT_ROOT: {root}")
print()

print("=== Top-level directories ===")
for d in sorted(os.listdir(root)):
    full = os.path.join(root, d)
    if os.path.isdir(full):
        has_model = os.path.isfile(os.path.join(full, "model.py"))
        print(f"  {d}/ {'(has model.py)' if has_model else ''}")

print()
print("=== Checking model imports ===")

dirs_to_try = [
    "glm5-kernels-flashmla-deepgemm",
    "glm5-kernels-flashinfer",
    "glm5-raw-decoupled-from-hf",
    "glm5-triton",
]

for dirname in dirs_to_try:
    full = os.path.join(root, dirname)
    if not os.path.isdir(full):
        print(f"  {dirname}: DIRECTORY NOT FOUND")
        continue

    model_file = os.path.join(full, "model.py")
    if not os.path.isfile(model_file):
        print(f"  {dirname}: no model.py")
        continue

    print(f"  {dirname}: model.py found, trying import...")
    sys.path.insert(0, full)
    try:
        import importlib
        spec = importlib.util.spec_from_file_location("model", model_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        classes = [x for x in dir(mod) if "Layer" in x or "Model" in x or "Attention" in x]
        print(f"    SUCCESS: {classes}")
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")
    finally:
        sys.path.pop(0)

print()
print("=== Done ===")
