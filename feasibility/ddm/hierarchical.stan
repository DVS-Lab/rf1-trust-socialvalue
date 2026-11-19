functions {
  real upper_logcdf(real y, real a, real t0, real z, real v) {
    if (y <= t0) return negative_infinity();
    return wiener_lcdf_unnorm(y, a, t0, z, v);
  }
  real retained_logmass(real a, real t0, real z, real v, real lo, real hi) {
    real high = log_sum_exp(upper_logcdf(hi,a,t0,z,v), upper_logcdf(hi,a,t0,1-z,-v));
    if (lo <= t0) return fmin(0,high);
    return log_diff_exp(high,log_sum_exp(upper_logcdf(lo,a,t0,z,v), upper_logcdf(lo,a,t0,1-z,-v)));
  }
  // Tail forms of the same first-passage density. Computing exp(-d^2/(2t))
  // before taking its log can poison the gradient of the contaminant mixture.
  real upper_logdensity(real rt, real a, real t0, real z, real v) {
    real t = rt - t0;
    real x = t / square(a);
    real w = 1 - z;
    if (x < 0.01) {
      // Leading image and its reflection. The next pair is suppressed by
      // at least exp(-2/x) relative to the first image for 0 < w < 1.
      real d = a * w;
      real reflected = log1p(2 * z / w) - 2 * z / x;
      return log(d) - log(t) + normal_lpdf(d | v * t, sqrt(t))
             + log1m_exp(reflected);
    }
    if (x > 10) {
      // Next spectral term has relative magnitude <= 4 exp(-1.5 pi^2 x).
      real sin_term = sin(pi() * fmin(w, z));
      return log(pi()) - 2 * log(a) + log(sin_term)
             - 0.5 * square(pi()) * x + v * a * w - 0.5 * square(v) * t;
    }
    return wiener_lpdf(rt | a,t0,z,v);
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
        // A zero target rejects the whole proposal, outside the mixture,
        // before either CDF is called. Both positive CDF times must be safe.
        if (is_inf(a) || is_nan(a) || a < 1e-6 || t0 >= hi
            || square(a)/(hi-t0) > 1000
            || (lo > t0 && square(a)/(lo-t0) > 1000)) return negative_infinity();
        for (k in 1:nc) {
          int c=cfirst[i]+k-1;
          int p=partner[c];
          int zi=7;
          if (Z == 3) zi=6+p;
          drift[k]=eta[i,2+p]+eta[i,6]*gap[c]+beta_zero*zero_offer[c];
          if (num_elements(beta_history) > 0)
            drift[k]+=dot_product(history[c],beta_history);
          bias[k]=inv_logit(eta[i,zi]+beta_side*side[c]);
          if (bias[k] <= 0 || bias[k] >= 1 || 1-bias[k] >= 1
              || is_nan(drift[k]) || abs(drift[k]) > 1e6) return negative_infinity();
          normalizer[k]=log_sum_exp(log1m(lapse)+retained_logmass(a,t0,bias[k],drift[k],lo,hi),log(lapse));
        }
        for (t in first[i]:last[i]) {
          int k=cell[t]-cfirst[i]+1;
          real ld=negative_infinity();
          if (rt[t] > t0) {
            if (y[t] == 1) ld=upper_logdensity(rt[t],a,t0,bias[k],drift[k]);
            else ld=upper_logdensity(rt[t],a,t0,1-bias[k],-drift[k]);
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
