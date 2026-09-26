functions {
  vector rl_logits(vector par, int model_code, int first, int last,
                   array[] int partner, vector gap, array[] int feedback,
                   vector outcome, vector ratings, array[] int zero_option, int zero_term) {
    vector[3] belief = rep_vector(0.5, 3);
    vector[last-first+1] logits;
    for (t in first:last) {
      int c = partner[t];
      real p = belief[c];
      real bonus = 0;
      real dv = gap[t] * (-1 + 1.5*p);
      if (model_code == 4) bonus = par[3]*ratings[c];
      if (model_code == 5 && c == 1) bonus = par[3];
      if (model_code == 7 && c < 3) bonus = par[c+2];
      dv += gap[t]*p*bonus;
      if (model_code == 9 && c < 3) dv += gap[t]*par[c+2];
      logits[t-first+1] = par[2]*dv;
      if (zero_term == 1) logits[t-first+1] += par[num_elements(par)]*zero_option[t];
      if (feedback[t] == 1) {
        real err = outcome[t]-p;
        real alpha = par[1];
        if (model_code == 8 && err < 0) alpha = par[3];
        belief[c] += alpha*err;
      }
    }
    return logits;
  }
}
data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=2,upper=5> K;
  int<lower=0,upper=2> A;
  int<lower=2,upper=9> model_code;
  int<lower=0,upper=1> bounded_theta;
  int<lower=0,upper=1> independent;
  array[K] int<lower=1,upper=4> kind;
  array[N] int<lower=1,upper=T> first;
  array[N] int<lower=1,upper=T> last;
  // Missing trials are removed here: they contribute neither likelihood nor updates.
  array[T] int<lower=1,upper=3> partner;
  vector<lower=0>[T] gap;
  array[T] int<lower=0,upper=1> y;
  array[T] int<lower=0,upper=1> feedback;
  vector<lower=0,upper=1>[T] outcome;
  matrix[N,3] ratings;
  matrix[N,A] age;
  vector[K] mu_location;
  vector<lower=0>[K] mu_scale;
  real<lower=0> prior_scale;
  int<lower=0,upper=1> prior_only;
  int<lower=0,upper=1> zero_term;
  array[T] int<lower=0,upper=1> zero_option;
}
parameters {
  vector[K] mu;
  vector<lower=0>[K] tau;
  matrix[K,A] beta;
  cholesky_factor_corr[K] L;
  matrix[K,N] z;
}
transformed parameters {
  matrix[N,K] eta;
  matrix[N,K] natural;
  matrix[K,N] random_effect;
  if (independent == 1) random_effect = diag_matrix(tau)*z;
  else random_effect = diag_pre_multiply(tau,L)*z;
  eta = rep_matrix(mu',N) + age*beta' + random_effect';
  for (i in 1:N) for (j in 1:K) {
    if (kind[j] == 1) natural[i,j] = inv_logit(eta[i,j]);
    else if (kind[j] == 2) natural[i,j] = exp(eta[i,j]);
    else if (kind[j] == 3) {
      if (bounded_theta == 1) natural[i,j] = 10*inv_logit(eta[i,j]);
      else natural[i,j] = log1p_exp(eta[i,j]);
    } else natural[i,j] = eta[i,j];
  }
}
model {
  mu ~ normal(mu_location, mu_scale*prior_scale);
  tau ~ normal(0, 0.8*prior_scale);
  to_vector(beta) ~ normal(0, 0.5*prior_scale);
  L ~ lkj_corr_cholesky(2);
  to_vector(z) ~ std_normal();
  if (prior_only == 0) {
    for (i in 1:N) {
      vector[last[i]-first[i]+1] logits = rl_logits(natural[i]',model_code,first[i],last[i],partner,gap,feedback,outcome,ratings[i]',zero_option,zero_term);
      y[first[i]:last[i]] ~ bernoulli_logit(logits);
    }
  }
}
generated quantities {
  corr_matrix[K] Omega = multiply_lower_tri_self_transpose(L);
  vector[N] log_lik;
  for (i in 1:N) {
    vector[last[i]-first[i]+1] logits = rl_logits(natural[i]',model_code,first[i],last[i],partner,gap,feedback,outcome,ratings[i]',zero_option,zero_term);
    log_lik[i] = bernoulli_logit_lpmf(y[first[i]:last[i]] | logits);
  }
}
