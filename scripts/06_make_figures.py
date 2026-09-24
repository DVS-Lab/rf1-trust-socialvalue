import json
from rf1_trust_socialvalue.behavior import compare_models
from rf1_trust_socialvalue.plotting import make_figures
from rf1_trust_socialvalue.reporting import report
from rf1_trust_socialvalue.gallery import build_gallery
if __name__ == '__main__':
    c=json.load(open('config/analysis.json'));compare_models(c);make_figures();report(c);build_gallery()
