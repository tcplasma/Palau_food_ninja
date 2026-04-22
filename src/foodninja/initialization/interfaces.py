from dataclasses import dataclass

from foodninja.core.models import SearchWindow


@dataclass(slots=True)
class InitializationResult:
    roi: SearchWindow
    histogram_ready: bool

