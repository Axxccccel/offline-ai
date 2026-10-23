# Runtime-hook: i fonsterlage (console=False) ar sys.stdout/stderr None.
# Nagot bibliotek som skriver dit kraschar da appen vid start. Peka om
# till os.devnull sa fonsterbygget startar lika stabilt som konsolbygget.
import sys, os
try:
    _null = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = _null
    if sys.stderr is None:
        sys.stderr = _null
except Exception:
    pass
