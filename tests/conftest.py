import os
import sys

# Make `src/` importable as top-level modules (e.g. `import features`)
# without needing to install this project as a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
