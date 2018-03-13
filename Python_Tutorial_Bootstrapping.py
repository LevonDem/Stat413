
import numpy as np

## Make up some data
n = 10000
X = np.random.normal(size = n)

## Calculate bootstrap confidence interval for the median
def BootstrapCI(X, B = 1000):

    boot_stat = np.zeros(B)
    for i in range(B):
        boot_indices = np.random.choice(range(n), n, replace = True)
        boot_sample = X[boot_indices]
        boot_stat[i] = np.median(boot_sample)

    CI_low  = np.percentile(boot_stat, q = 0.5) 
    CI_high = np.percentile(boot_stat, q = 99.5)
    CI      = [CI_low, CI_high]    
    return(CI)

## Calculate 'true' confidence interval for the median
def TrueCI(X):
    
    n = len(X)
    low     = int(round((n/2) - (2.54 * np.sqrt(n) / 2)))
    high    = int(round(1 + (n/2) + (2.54 * np.sqrt(n) / 2)))
    CI_true = [np.sort(X)[low], np.sort(X)[high]]
    return(CI_true)

## Compare results using both methods
print(BootstrapCI(X, B = 1000))
print(TrueCI(X))

## Plot the results
import matplotlib.pyplot as plt

#n, bins, patches = plt.hist(X, 50, normed = 1, facecolor = 'green', alpha = 0.75)
plt.hist(X, 50,  facecolor = 'green', alpha = 0.75)
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('$\mathrm{Histogram\ of\ Normal(0,1)\ random\ variables:}\ \mu=0,\ \sigma=1$')
plt.axis([-4, 4, 0, 1000])
plt.grid(True)

# Add CIs
low_true, high_true = TrueCI(X)
plt.axvline(x = low_true, linestyle = '--', linewidth = 0.5)
plt.axvline(x = high_true, linestyle = '--', linewidth = 0.5)

plt.show()

