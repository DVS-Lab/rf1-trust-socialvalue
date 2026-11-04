"""Short synthetic execution check; explicitly not a scientific posterior fit."""
from pathlib import Path
import time
import numpy as np
from hierarchical import config,compile_model,synthetic_fixture,model_data,initial_values,diagnostics
from hier_simulation import posterior_arrays,participant_posterior,simulate,ppc_summary
from hier_data import WORK,save,source_hashes


def main():
    c=config();c=dict(c,warmup=30,draws=20,max_treedepth=8,adapt_delta=.9)
    t=synthetic_fixture(2);t=t[t.trial_in_run.le(9)].copy()
    data,meta=model_data(t,c,"partner")
    folder=WORK/"engineering_smoke"/str(int(time.time()));folder.mkdir(parents=True,exist_ok=True)
    model=compile_model(c)
    fit=model.sample(data=data,chains=4,parallel_chains=4,threads_per_chain=1,force_one_process_per_chain=True,
        iter_warmup=30,iter_sampling=20,adapt_delta=.9,max_treedepth=8,
        seed=c["seed"],inits=[initial_values(data,c["seed"]+i) for i in range(4)],
        output_dir=str(folder),show_progress=False,refresh=10)
    quality,summary=diagnostics(fit,c);draws=posterior_arrays(fit)
    p=participant_posterior(draws,meta["ids"])
    assert len(p)==2*10 and np.isfinite(p[["mean","sd","q025","q975"]]).all().all()
    for conditional in [False,True]:
        simulated=simulate(t,draws["eta"][0],draws["beta_zero"][0],draws["beta_side"][0],draws["beta_history"][0],
            c,3,31,conditional=conditional)
        cells=ppc_summary(simulated)
        assert len(cells) and set(cells.cell_kind)=={"partner","partner_offer","partner_choice"}
    status=dict(status="synthetic_execution_passed",subjects=2,chains=4,warmup_per_chain=30,draws_per_chain=20,
        scientific_posterior_validation=False,note="Short execution/shape check only; diagnostics are not a scientific acceptance test.",
        csv_reading=True,parameter_extraction=True,posterior_simulation=True,sources=source_hashes())
    save(WORK/"engineering_smoke/status.json",status)
    print("Synthetic execution, CSV diagnostics, parameter extraction and posterior simulation passed.")


if __name__=="__main__":main()
