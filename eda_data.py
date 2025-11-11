import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport



df = pd.read_excel("data/data.xlsx")

df.info()
profile = ProfileReport(df)
profile.to_file("ad_real_estate_report.html")