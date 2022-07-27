import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from relife.datasets import load_circuit_breaker
from relife import KaplanMeier, Weibull, Gompertz, AgeReplacementPolicy

print("i'm alive")

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')
sample_file_name = "sample"

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

time, event, entry = load_circuit_breaker().astuple()
km = KaplanMeier().fit(time, event, entry)
weibull = Weibull().fit(time, event, entry)
gompertz = Gompertz().fit(time, event, entry)

km.plot()
weibull.plot()
gompertz.plot()
plt.xlabel('Age [year]')
plt.ylabel('Survival probability')

#keep this for both 
# a0 = np.array([15, 20, 25]).reshape(-1,1)
# cp = 10
# cf = np.array([900, 500, 100]).reshape(-1,1)
# policy = AgeReplacementPolicy(gompertz, a0=a0, cf=cf, cp=cp, rate=0.04)
# policy.fit()
# policy.ar1, policy.ar

# a = np.arange(1,100,0.1)
# za = policy.asymptotic_expected_equivalent_annual_cost(a)
# za_opt = policy.asymptotic_expected_equivalent_annual_cost()
# plt.plot(a, za.T)
# for i, ar in enumerate(policy.ar):
#     plt.scatter(ar, za_opt[i], c=f'C{i}',
#         label=f" cf={cf[i,0]} k€, ar={ar[0]:0.1f} years")
# plt.xlabel('Age of preventive replacement [years]')
# plt.ylabel('Asymptotic expected equivalent annual cost [k€]')
# plt.legend()

# dt = 0.5
# step = int(1/dt)
# t = np.arange(0, 30+dt, dt)
# z = policy.expected_total_cost(t).sum(axis=0)
# y = t[::step][1:]
# q = np.diff(z[::step])
# plt.bar(2020+y, q, align='edge', width=-0.8, alpha=0.8, color='C2')
# plt.xlabel('Year')
# plt.ylabel('Expected discounted annual cost in k€')

# mt = policy.expected_total_cost(t, cf=1, cp=1, rate=0).sum(axis=0)
# mf = policy.expected_total_cost(t, cf=1, cp=0, rate=0).sum(axis=0)
# qt = np.diff(mt[::step])
# qf = np.diff(mf[::step])
# plt.bar(y+2020, qt, align='edge', width=-0.8, alpha=0.8,
#     color='C1', label='all replacements')
# plt.bar(y+2020, qf, align='edge', width=-0.8, alpha=0.8,
#     color='C0', label='failure replacements only')
# plt.xlabel('Years')
# plt.ylabel('Expected number of annual replacements')
# plt.legend()

plt.savefig(results_dir + sample_file_name)
