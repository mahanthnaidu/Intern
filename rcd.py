import pandas as pd
from pyrca.analyzers.rcd import *
normal_df = pd.read_csv('normal.csv')
normal_df = normal_df.iloc[:,1:]
config = RCDConfig(start_alpha=0.01, alpha_step=0.1, alpha_limit=1, localized=True, gamma=5, bins=5, k=3, f_node='F-node', verbose=False, ci_test='chisq')
rcd_rca = RCD(config)
abnormal_df = pd.read_csv('abnormal.csv')
root_causes = rcd_rca.find_root_causes(normal_df, abnormal_df)
print("RCD RCA Root Causes:", root_causes)
