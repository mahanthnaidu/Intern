import pandas as pd
from pyrca.graphs.causal import *
from pyrca.analyzers.random_walk import *
df = pd.read_csv('normal_data.csv')
fges_config = FGESConfig(domain_knowledge_file=None, run_pdag2dag=True, max_num_points=5000000, max_degree=10, penalty_discount=80, score_id='sem_bic_score')
fges_algo = FGES(fges_config)
graph_df = fges_algo.train(df)
abnormal_df = pd.read_csv('abnormal_data.csv')
config = RandomWalkConfig(graph=graph_df)
rw_rca = RandomWalk(config)
anomalous_metrics = ['Monthly Sales']  
root_causes = rw_rca.find_root_causes(anomalous_metrics,df=None)
print("Random Walk RCA Root Causes:", root_causes)