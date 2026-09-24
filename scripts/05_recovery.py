import json
from rf1_trust_socialvalue.recovery import parameter_recovery, model_recovery, predictive_checks
if __name__ == '__main__':
    c=json.load(open('config/analysis.json'))
    parameter_recovery(c)
    model_recovery(c)
    predictive_checks(c)
