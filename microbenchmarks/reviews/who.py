import pandas as pd
import pandas as pl
import numpy as np
import platform
import pyarrow
import os


def diagnose_me():
  print(f"Python {platform.python_version()}")
  for (name, pkg) in [("Pandas", pd), ("NumPy", np), ("Polars", pl),
                      ("PyArrow", pyarrow)]:
    print(f"{name}: {pkg.__version__}")
  pmt = "unset"
  if "POLARS_MAX_THREADS" in os.environ:
    pmt = os.environ["POLARS_MAX_THREADS"]
  print(f"POLARS_MAX_THREADS={pmt}")


diagnose_me()
