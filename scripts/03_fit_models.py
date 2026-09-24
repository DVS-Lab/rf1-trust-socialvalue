import json
from rf1_trust_socialvalue.fitting import run_fits
if __name__ == '__main__':
    config=json.load(open('config/analysis.json'))
    run_fits(config)
    run_fits(config,heldout=True)
