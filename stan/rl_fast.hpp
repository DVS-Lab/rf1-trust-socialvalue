// Exact first derivatives for the sequential likelihood, registered on Stan's
// reverse-mode tape as one operation per participant. No approximation to NUTS.
#include <stan/math.hpp>
#include <array>
#include <vector>
#include <cmath>

template <bool propto, typename TP, typename TG, typename TO, typename TR>
stan::return_type_t<TP> rl_fast_lpmf(
    const std::vector<int>& y, const TP& pars, const int& model_code,
    const int& first, const int& last, const std::vector<int>& partner,
    const TG& gap, const std::vector<int>& feedback, const TO& outcome,
    const TR& ratings, std::ostream* pstream) {
  const int K = pars.size();
  std::array<double,4> par{}, grad{};
  for (int j=0;j<K;++j) par[j]=stan::math::value_of(pars(j));
  std::array<double,3> belief{0.5,0.5,0.5};
  std::array<std::array<double,4>,3> deriv{};
  double ll=0;
  for (int t=first-1;t<last;++t) {
    const int c=partner[t]-1;
    const double p=belief[c], g=gap(t);
    double bonus=0, pref=0;
    std::array<double,4> db{}, dpref{};
    if (model_code==4) { bonus=par[2]*ratings(c); db[2]=ratings(c); }
    if (model_code==5 && c==0) { bonus=par[2]; db[2]=1; }
    if (model_code==7 && c<2) { bonus=par[c+2]; db[c+2]=1; }
    if (model_code==9 && c<2) { pref=par[c+2]; dpref[c+2]=1; }
    const double dv=g*(-1+1.5*p+p*bonus+pref);
    const double z=par[1]*dv;
    const double probability=stan::math::inv_logit(z);
    ll += y[t]*z-stan::math::log1p_exp(z);
    for (int j=0;j<K;++j) {
      const double dz=par[1]*g*((1.5+bonus)*deriv[c][j]+p*db[j]+dpref[j])+(j==1?dv:0);
      grad[j]+=(y[t]-probability)*dz;
    }
    if (feedback[t]) {
      const double err=outcome(t)-p;
      const int a=(model_code==8 && err<0)?2:0;
      for (int j=0;j<K;++j) deriv[c][j]=(1-par[a])*deriv[c][j]+(j==a?err:0);
      belief[c]+=par[a]*err;
    }
  }
  if constexpr (stan::is_var<stan::scalar_type_t<TP>>::value) {
    std::vector<stan::math::var> operands;
    std::vector<double> gradients;
    for (int j=0;j<K;++j) { operands.push_back(pars(j)); gradients.push_back(grad[j]); }
    return stan::math::precomputed_gradients(ll,operands,gradients);
  } else {
    return ll;
  }
}
