import pandas as pd
from pyrca.graphs.causal import *
from pyrca.analyzers.ht import *
df = pd.read_csv('normal.csv')
pc_config = PCConfig(domain_knowledge_file=None, run_pdag2dag=True, max_num_points=5000000, alpha=0.01)
pc_algo = PC(pc_config)
graph_df = pc_algo.train(df)
config = HTConfig(graph=graph_df, aggregator='max', root_cause_top_k=3)
ht_rca = HT(config)
normal_df = pd.read_csv('normal.csv')
ht_rca.train(normal_df)
abnormal_df = pd.DataFrame('abnormal.csv')
root_causes = ht_rca.find_root_causes(abnormal_df)
print("HT RCA Root Causes:", root_causes)