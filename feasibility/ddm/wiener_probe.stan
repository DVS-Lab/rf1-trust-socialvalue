data {
  int<lower=1> M;
  vector[M] rt;
  vector[M] a;
  vector[M] t0;
  vector[M] z;
  vector[M] drift;
}
generated quantities {
  vector[M] log_density;
  vector[M] log_cdf;
  for (m in 1:M) {
    log_density[m]=wiener_lpdf(rt[m] | a[m],t0[m],z[m],drift[m]);
    log_cdf[m]=wiener_lcdf_unnorm(rt[m],a[m],t0[m],z[m],drift[m]);
  }
}
