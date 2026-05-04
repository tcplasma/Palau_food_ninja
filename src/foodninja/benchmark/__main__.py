"""Allows ``python -m foodninja.benchmark`` to invoke the CLI runner."""
from foodninja.benchmark.runner import main
import sys

sys.exit(main())
