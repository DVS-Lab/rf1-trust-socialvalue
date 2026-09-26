# N=111 closeout scope — 25 September 2026

This document records the user-adopted closeout brief. It is a finite developmental analysis of the current ds005123 v1.1.3 primary N=111 sample. The additional approximately 235 participants must not be accessed, inspected, imported, summarized, or analyzed. There are no instructions here for importing them.

## A. Diagnose before extending

Preserve the accepted-fit report and all posterior caches. Continue to exclude H4_full_age and H5_train_age; no further automatic sampling escalation for either.

For accepted no-age H2/H5/H7/HPreference/H8, compare observed-history conditional prediction with fully generative prediction. Existing accepted no-age posteriors are TRAINING posteriors, so stage A uses them unchanged and labels all/training/held-out periods. No full-data no-age posteriors are invented or silently substituted with age fits. Published full-age PPCs serve only to verify the motivating pattern.

Use actual offers, partner order, programmed outcomes, and missed-trial patterns. Conditional predictions update from actual observed feedback only. Generative predictions simulate choices and reveal feedback only after simulated positive investments. Stratify primarily by partner × exact offer pair, then marginally by zero-option status, actual run, chronological thirds, and previous same-partner observed feedback. Do not cross every feature into sparse cells. Use equal participant weighting within cells and flag sparse cells. Keep history grouping conventions explicit.

Verify the zero-option examples directly from committed outputs. Interpret them as a possible positive-investment bias, not evidence for a specific social-reward mechanism.

## B. Only one conditional extension

If the diagnostic supports it, add a single generic term to the choice logit:

`logit(P(high)) = existing_logit + gamma0 * I(low_option == 0)`.

The participant-level gamma0 parameter is shared across partners and partially pooled. It is not multiplied by kappa. Use a weakly informative prior and otherwise preserve each corresponding model's priors and pooling conventions. Fit no-age H5, H7, HPreference, and H8 with this feature. Make no partner-specific zero term and no second extension.

Compare full-data no-age base/extended PPCs on zero-containing and positive-positive offers, paired temporal held-out log loss and Brier score, and changes in theta/preference, alpha, and kappa. Age-fit versus no-age differences must not masquerade as effects of gamma0. A matched full-data no-age baseline may therefore require new fits in this stage. Accuracy is secondary. Carry the feature forward only if it repairs relevant PPCs and improves held-out performance consistently across the leading partner models; descriptive improvement alone is insufficient. Record rejection or inconclusive evidence honestly.

## C. Targeted realistic recovery

Use H5/H7/HPreference/H8 as the central generating and fitting set; H2 is optional. If the zero feature earns retention, use adjusted variants primarily and original variants secondarily. Generate from empirical posterior parameter ranges using actual N=111 task schedules and missingness. Cover moderate theta, 5–10, and the upper empirical tail, adding an explicit high-theta stress condition if posterior resampling is inadequate above about 8–10. Never truncate generating theta at 5.

Begin with a fast nonhierarchical screen with generous starts and at least theta≤20 fitting bounds, checking whether any upper-tail generators exceed the fitting range. Fit every candidate to every generated dataset; report AICc/BIC and temporal prediction where practical. Show actual H7↔HPreference selection rates and confusion with H8. Only if this screen suggests useful discrimination should a small 3–5 datasets per generating mechanism hierarchical confirmation be run. Strong confusion is a result, not a reason to launch many more Stan fits. Do not treat one arbitrary selection cutoff as proof of identifiability.

## D. Freeze age and ratings scope

No new age forms, interactions, subgroups, age bins for inference, or extensive age power simulations. Retain behavioral age-25-to-75 effects and the accepted H5 age effect with uncertainty; latent variance fractions are not total behavioral variance explained. No substantial ratings investigation. Timing remains unresolved unless an immediately available documentary source settles it. H4/M3 remain secondary.

## E. Final artifacts and stopping point

The authoritative endpoint will be `results/n111_wrapup/README.md` once all decisions are resolved. It must answer the five questions about robust behavior, hierarchical evidence, mechanistic ambiguity, the PPC failure, and age uncertainty, ending with “What should carry forward to the full dataset”. Maintain `docs/n111_wrapup_decisions.md` as the decision record.

Required final tables:

- predictive_residuals.csv
- conditional_vs_generative.csv
- zero_option_model_comparison.csv
- zero_option_parameter_summary.csv
- realistic_model_recovery.csv
- realistic_model_recovery_confusion.csv
- n111_conclusions.csv

Final figures (PNG, vector PDF and SVG): where models miss; conditional versus generative residuals; base versus zero-option PPC/held-out comparison if fitted; targeted recovery confusion; compact N=111 synthesis. No large extra gallery.

Stop once the zero-option localization, feature retention/rejection, H7-versus-HPreference recoverability in empirical parameter ranges, and existing uncertain age conclusion are recorded. Do not add utility functions, partner parameters, learning variants, age forms, sampler tuning, or rating models beyond this scope. Partial stage outputs must never be labeled “N=111 closeout complete”.

Before commits run pytest, scripts/07_validate_outputs.py, and relevant closeout validation; inspect staged files for raw/imaging content. Preserve strict hierarchical acceptance criteria. Commit logical stages and push main without force. Linux results and console/error logs are committed by the user, who controls linux1.
