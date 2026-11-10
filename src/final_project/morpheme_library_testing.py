from pathlib import Path

from morphemes import Morphemes

# Resolve relative to the project root (not the current working directory),
# and make sure it exists: morphemes writes <path>/config.json but never creates the folder.
data_dir = Path(__file__).resolve().parents[2] / "data"
data_dir.mkdir(parents=True, exist_ok=True)
path = str(data_dir)

m = Morphemes(path) #Data path is optional, local storage will be used if left out.
print(m.parse("organizationally"))
