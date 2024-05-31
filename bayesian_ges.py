import pandas as pd
from pyrca.graphs.causal import *
from pyrca.analyzers.bayesian import *
df = pd.read_csv('normal.csv')
ges_config = GESConfig(domain_knowledge_file=None, run_pdag2dag=True, max_num_points=5000000, max_degree=5, penalty_discount=100)
ges_algo = GES(ges_config)
graph_df = ges_algo.train(df)
config = BayesianNetworkConfig(graph=graph_df, sigmas=None, default_sigma=4.0, thres_win_size=5, thres_reduce_func='mean', infer_method='posterior', root_cause_top_k=3)
bn_rca = BayesianNetwork(config)
anomalous_metrics = ['Business Metric']  
root_causes = bn_rca.find_root_causes(anomalous_metrics, set_zero_path_score_for_normal_metrics=False)
print("Bayesian RCA Root Causes:", root_causes)