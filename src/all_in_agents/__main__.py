"""Allow running as ``python -m all_in_agents``."""
from .cli import main
import sys

sys.exit(main())
