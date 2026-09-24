import json
from rf1_trust_socialvalue.behavior import run_behavior
if __name__ == '__main__':run_behavior(json.load(open('config/analysis.json')))
