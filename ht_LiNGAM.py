import pandas as pd
from pyrca.graphs.causal import *
from pyrca.analyzers.ht import *
df = pd.read_csv('normal.csv')
lingam_config = LiNGAMConfig(domain_knowledge_file=None, run_pdag2dag=True, max_num_points=5000000, lower_limit=0.1, n_sampling=-1, min_causal_effect=0.01)
lingam_algo = LiNGAM(lingam_config)
graph_df = lingam_algo.train(df)
config = HTConfig(graph=graph_df, aggregator='max', root_cause_top_k=3)
ht_rca = HT(config)
normal_df = pd.read_csv('normal.csv')
ht_rca.train(normal_df)
abnormal_df = pd.DataFrame('abnormal.csv')
root_causes = ht_rca.find_root_causes(abnormal_df)
print("HT RCA Root Causes:", root_causes)