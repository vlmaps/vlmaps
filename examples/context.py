from os.path import abspath, dirname, join
import sys

print("__file__: ", __file__)
parent_dir = abspath(join(dirname(__file__), ".."))
print(f"imported path: {parent_dir}")
sys.path.insert(0, parent_dir)
