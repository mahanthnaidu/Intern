import pandas as pd
from pyrca.analyzers.epsilon_diagnosis import *
normal_df = pd.read_csv('normal.csv')
config = EpsilonDiagnosisConfig(alpha=0.05, bootstrap_time=200, root_cause_top_k=3)
ed_rca = EpsilonDiagnosis(config)
ed_rca.train(normal_df)
abnormal_df = pd.read_csv('abnormal.csv') 
root_causes = ed_rca.find_root_causes(abnormal_df)
print("Epsilon Diagnosis RCA Root Causes:", root_causes)
