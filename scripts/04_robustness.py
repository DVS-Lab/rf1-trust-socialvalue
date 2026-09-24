import json
from rf1_trust_socialvalue.robustness import run_robustness
if __name__ == '__main__':run_robustness(json.load(open('config/analysis.json')))
