import pandas as pd
from pyrca.graphs.causal import *
from pyrca.analyzers.random_walk import *
df = pd.read_csv('normal.csv')
lingam_config = LiNGAMConfig(domain_knowledge_file=None, run_pdag2dag=True, max_num_points=5000000, lower_limit=0.1, n_sampling=-1, min_causal_effect=0.01)
lingam_algo = LiNGAM(lingam_config)
graph_df = lingam_algo.train(df)
abnormal_df = pd.read_csv('abnormal.csv')
config = RandomWalkConfig(graph=graph_df)
rw_rca = RandomWalk(config)
anomalous_metrics = ['Business Metric']  
root_causes = rw_rca.find_root_causes(anomalous_metrics,df=None)
print("Random Walk RCA Root Causes:", root_causes)