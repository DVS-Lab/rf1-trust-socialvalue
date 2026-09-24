import argparse
from rf1_trust_socialvalue.hierarchical import run_one,prior_predictive
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',default='H5');p.add_argument('--train',action='store_true');p.add_argument('--age-terms',type=int,default=1);p.add_argument('--bounded',action='store_true');p.add_argument('--prior',action='store_true');p.add_argument('--prior-scale',type=float,default=1);a=p.parse_args()
 if a.prior: prior_predictive(a.model)
 else: run_one(a.model,a.train,a.age_terms,a.bounded,a.prior_scale)
