import matplotlib
matplotlib.use('Agg') #prevent error running remotely
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
# plt.plot(range(20), range(20))
xp = np.linspace(1, 60, 200)
# _ = plt.plot(xp, [np.log(2**x) for x in xp], '-')
# _ = plt.plot([np.log(x) for x in xp], [np.log(np.exp(np.log(x)**1.4 + np.log(x))) for x in xp], '--')
# _ = plt.plot([np.log(x) for x in xp], [np.log(np.exp(np.log(x)**1.4 + np.log(x))) for x in xp], '--')
_ = plt.plot([np.log(x) for x in xp], [np.log(x**3) for x in xp], '--')
# _ = plt.plot([x for x in xp], [np.log(x**2) for x in xp], '--')

plt.show()
fig.savefig('test plot', bbox_inches='tight')    
plt.close()

