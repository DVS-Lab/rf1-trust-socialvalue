import json
from rf1_trust_socialvalue.age_behavior import run
if __name__ == "__main__":
    run(json.load(open("config/analysis.json")))
