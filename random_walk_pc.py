import pandas as pd
from pyrca.graphs.causal import *
from pyrca.analyzers.random_walk import *
df = pd.read_csv('normal.csv')
pc_config = PCConfig(domain_knowledge_file=None, run_pdag2dag=True, max_num_points=5000000, alpha=0.01)
pc_algo = PC(pc_config)
graph_df = pc_algo.train(df)
abnormal_df = pd.read_csv('abnormal.csv')
config = RandomWalkConfig(graph=graph_df)
rw_rca = RandomWalk(config)
anomalous_metrics = ['Business Metric']  
root_causes = rw_rca.find_root_causes(anomalous_metrics,df=None)
print("Random Walk RCA Root Causes:", root_causes)