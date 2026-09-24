import argparse,json
from rf1_trust_socialvalue.bounds import bound_fits,bound_recovery,figures
if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--recovery",action="store_true");a=p.parse_args();c=json.load(open("config/analysis.json"))
    if a.recovery: bound_recovery(c)
    else: bound_fits(c);figures()
