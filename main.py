import numpy as np
from scipy.stats import beta, norm
 
class BayesianDecisionModel:
    def __init__(self, prior_alpha=1, prior_beta=1):
        self.alpha=prior_alpha; self.beta_param=prior_beta
 
    def update(self, successes, failures):
        self.alpha+=successes; self.beta_param+=failures
 
    def posterior_mean(self): return self.alpha/(self.alpha+self.beta_param)
 
    def credible_interval(self, ci=0.95):
        return beta.ppf([(1-ci)/2, 1-(1-ci)/2], self.alpha, self.beta_param)
 
    def expected_utility(self, choices, payoffs):
        p=self.posterior_mean()
        return {c: sum(payoff*p**i*(1-p)**(1-i) for i,(payoff,i) in enumerate(zip(payoffs,[1,0])))
                for c in choices}
 
class SignalDetectionModel:
    def __init__(self, d_prime=2.0, criterion=0.0):
        self.d=d_prime; self.c=criterion
    def hit_rate(self): return 1-norm.cdf(self.c-self.d/2)
    def false_alarm_rate(self): return 1-norm.cdf(self.c+self.d/2)
    def sensitivity_roc(self, criteria):
        return [(1-norm.cdf(c+self.d/2), 1-norm.cdf(c-self.d/2)) for c in criteria]
 
model=BayesianDecisionModel(1,1)
model.update(7,3)
print(f"Posterior mean: {model.posterior_mean():.3f}")
print(f"95% CI: {model.credible_interval()}")
sdm=SignalDetectionModel(d_prime=2.5); print(f"Hit rate: {sdm.hit_rate():.3f}, FA: {sdm.false_alarm_rate():.3f}")
