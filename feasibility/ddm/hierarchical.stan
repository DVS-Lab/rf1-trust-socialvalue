functions {
  real upper_logcdf(real y, real a, real t0, real z, real v) {
    if (y <= t0) return negative_infinity();
    // Extreme warmup proposals can make both native CDF series lengths infinite,
    // producing a nonterminating floating-point loop in Stan Math 2.40.
    // This generous numerical domain is far beyond the pilot's fitting bounds.
    if (is_inf(a) || is_nan(a) || a<1e-6 || square(a)/(y-t0)>1000 || abs(v)>1e6)
      reject("Wiener numerical evaluation domain exceeded");
    return wiener_lcdf_unnorm(y, a, t0, z, v);
  }
  real retained_logmass(real a, real t0, real z, real v, real lo, real hi) {
    real high = log_sum_exp(upper_logcdf(hi,a,t0,z,v), upper_logcdf(hi,a,t0,1-z,-v));
    if (lo <= t0) return fmin(0,high);
    return log_diff_exp(high,log_sum_exp(upper_logcdf(lo,a,t0,z,v), upper_logcdf(lo,a,t0,1-z,-v)));
  }
  real decision_logdensity(real rt, int high_choice, real a, real t0, real z, real v,
                           real lo, real hi, real lapse) {
    real ld = negative_infinity();
    real lm = retained_logmass(a,t0,z,v,lo,hi);
    if (rt > t0) {
      if (high_choice == 1) ld = wiener_lpdf(rt | a,t0,z,v);
      else ld = wiener_lpdf(rt | a,t0,1-z,-v);
    }
    // One unconditional mixture, then condition both choices jointly on RT retention.
    return log_mix(lapse,-log(2*(hi-lo)),ld)-log_sum_exp(log1m(lapse)+lm,log(lapse));
  }
  real partial_likelihood(array[] int persons, int start, int end,
                          matrix eta, real beta_zero, real beta_side, vector beta_history,
                          int Z, array[] int first, array[] int last,
                          array[] int cfirst, array[] int clast,
                          array[] int cell, array[] int partner, vector gap, vector zero_offer,
                          vector side, matrix history, vector rt, array[] int y,
                          real lo, real hi, real lapse) {
    real lp=0;
    for (ii in 1:size(persons)) {
      int i=persons[ii];
      int nc=clast[i]-cfirst[i]+1;
      real a=exp(eta[i,1]);
      real t0=hi*inv_logit(eta[i,2]);
      vector[nc] drift;
      vector[nc] bias;
      vector[nc] normalizer;
      if (nc > 0) {
        for (k in 1:nc) {
          int c=cfirst[i]+k-1;
          int p=partner[c];
          int zi=7;
          if (Z == 3) zi=6+p;
          drift[k]=eta[i,2+p]+eta[i,6]*gap[c]+beta_zero*zero_offer[c]+dot_product(history[c],beta_history);
          bias[k]=inv_logit(eta[i,zi]+beta_side*side[c]);
          normalizer[k]=log_sum_exp(log1m(lapse)+retained_logmass(a,t0,bias[k],drift[k],lo,hi),log(lapse));
        }
        for (t in first[i]:last[i]) {
          int k=cell[t]-cfirst[i]+1;
          real ld=negative_infinity();
          if (rt[t] > t0) {
            if (y[t] == 1) ld=wiener_lpdf(rt[t] | a,t0,bias[k],drift[k]);
            else ld=wiener_lpdf(rt[t] | a,t0,1-bias[k],-drift[k]);
          }
          lp+=log_mix(lapse,-log(2*(hi-lo)),ld)-normalizer[k];
        }
      }
    }
    return lp;
  }
}
data {
  int<lower=1> N;
  int<lower=0> T;
  int<lower=0> C;
  int<lower=1,upper=3> Z;
  int<lower=0,upper=2> H;
  int<lower=1> grainsize;
  array[N] int<lower=1,upper=N> person;
  array[N] int<lower=1> first;
  array[N] int<lower=0> last;
  array[N] int<lower=1> cfirst;
  array[N] int<lower=0> clast;
  array[T] int<lower=1,upper=C> cell;
  array[C] int<lower=1,upper=3> partner;
  vector[C] gap;
  vector[C] zero_offer;
  vector[C] side;
  matrix[C,H] history;
  vector<lower=0>[T] rt;
  array[T] int<lower=0,upper=1> y;
  real<lower=0> lo;
  real<lower=lo> hi;
  real<lower=0,upper=1> lapse;
  vector[6+Z] mu_location;
  vector<lower=0>[6+Z] mu_scale;
  vector<lower=0>[6+Z] tau_scale;
  real<lower=0> prior_scale;
  int<lower=0,upper=1> prior_only;
  int<lower=0,upper=1> likelihood_only;
}
parameters {
  vector[6+Z] mu;
  vector<lower=0>[6+Z] tau;
  cholesky_factor_corr[6+Z] L_Omega;
  matrix[6+Z,N] raw;
  real beta_zero;
  real beta_side;
  vector[H] beta_history;
}
transformed parameters {
  matrix[N,6+Z] eta=(rep_matrix(mu,N)+diag_pre_multiply(tau,L_Omega)*raw)';
  vector<lower=0>[N] a;
  vector<lower=0,upper=hi>[N] t0;
  matrix<lower=0,upper=1>[N,3] z;
  vector[N] z_difference;
  for (i in 1:N) {
    a[i]=exp(eta[i,1]);
    t0[i]=hi*inv_logit(eta[i,2]);
    for (p in 1:3) {
      if (Z==1) z[i,p]=inv_logit(eta[i,7]);
      else z[i,p]=inv_logit(eta[i,6+p]);
    }
    z_difference[i]=z[i,1]-z[i,2];
  }
}
model {
  if (likelihood_only==0) {
    mu ~ normal(mu_location,mu_scale*prior_scale);
    tau ~ normal(0,tau_scale*prior_scale);
    L_Omega ~ lkj_corr_cholesky(2);
    to_vector(raw) ~ std_normal();
    beta_zero ~ normal(0,prior_scale);
    beta_side ~ normal(0,0.5*prior_scale);
    beta_history ~ normal(0,0.5*prior_scale);
  }
  if (prior_only==0)
    target+=reduce_sum(partial_likelihood,person,grainsize,eta,beta_zero,beta_side,beta_history,
                       Z,first,last,cfirst,clast,cell,partner,gap,zero_offer,side,history,rt,y,lo,hi,lapse);
}
generated quantities {
  corr_matrix[6+Z] Omega=multiply_lower_tri_self_transpose(L_Omega);
  real cohort_mean_z_difference=mean(z_difference);
  real cohort_sd_z_difference=sd(z_difference);
}
