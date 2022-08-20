#!/usr/bin/env python
# coding: utf-8

# # Q7

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


cars=pd.read_csv("Q7.csv")


# In[29]:


cars


# In[30]:


cars.mean()


# In[31]:


cars.median()


# In[32]:


cars.Points.mode() 


# In[33]:


cars.Score.mode()


# In[34]:


cars.Weigh.mode()


# In[35]:


cars.var()


# In[36]:


cars.std()


# In[37]:


cars.describe()


# In[38]:


Points_Range=cars.Points.max()-cars.Points.min()
Points_Range


# In[39]:


Score_Range=cars.Score.max()-cars.Score.min()
Score_Range


# In[40]:


Weigh_Range=cars.Weigh.max()-cars.Weigh.min()
Weigh_Range


# In[41]:


f,ax=plt.subplots(figsize=(15,5))
plt.subplot(1,3,1)
plt.boxplot(cars.Points)
plt.title('Points')
plt.subplot(1,3,2)
plt.boxplot(cars.Score)
plt.title('Score')
plt.subplot(1,3,3)
plt.boxplot(cars.Weigh)
plt.title('Weigh')
plt.show()


# # Q9A and Q9B

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis
from scipy.stats import skew
import seaborn as sns


# In[43]:


Q9A_data = pd.read_csv("Q9_a.csv")
Q9B_data = pd.read_csv("Q9_b.csv")
Q9A = pd.DataFrame(Q9A_data)
print(Q9A.head())


# In[44]:


print ('\n------------Kurtosis for Normal Distribution--------\n', Q9A_data.kurt())
print(kurtosis(Q9A_data))
#plt.plot(Q9A_data)
#plt.show()

sns.distplot(kurtosis(Q9A_data), label = 'Kurtosis')
plt.xlabel('speed')
plt.ylabel('distance')
plt.legend()


# In[45]:


print ('\n------------Skewness for Normal Distribution--------\n', Q9A_data.skew())
print(skew(Q9A_data))
#plt.plot(Q9A_data)
#plt.show()

sns.distplot(skew(Q9A_data), label = 'Skewness')
plt.xlabel('speed')
plt.ylabel('distance')
plt.legend()


# In[46]:


Q9B_data = pd.read_csv("Q9_b.csv")
Q9B = pd.DataFrame(Q9B_data)
print(Q9B.head())


# In[47]:


print ('\n------------Kurtosis for Normal Distribution--------\n', Q9B_data.kurt())
print(kurtosis(Q9B_data))
#plt.plot(Q9B_data)
#plt.show()

sns.distplot(kurtosis(Q9B_data), label = 'Kurtosis')
plt.xlabel('SP')
plt.ylabel('WT')
plt.legend()


# In[48]:


print ('\n------------Skewness for Normal Distribution--------\n', Q9B_data.skew())
print(skew(Q9B_data))
#plt.plot(Q9B_data)
#plt.show()

sns.distplot(skew(Q9B_data), label = 'Skewness')
plt.xlabel('SP')
plt.ylabel('WT')
plt.legend()


# # Q20

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis
from scipy.stats import skew
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


Q20_data = pd.read_csv("Cars.csv")
Q20 = pd.DataFrame(Q20_data)
print(Q20.head())
print(Q20.tail())


# In[51]:


Q20.describe()


# In[52]:


print ('\n------------Probability of MPG of Cars P(MPG>38)--------\n')
Q20_data[Q20.MPG>38]


# In[53]:


print ('\n------------Probability of MPG of Cars P(MPG<40)--------\n')
Q20_data[Q20.MPG<40]


# In[54]:


print ('\n------------Probability of MPG of Cars P(20<MPG<50)--------\n')
Q20_data[(Q20.MPG>20) & (Q20.MPG<50)]


# In[55]:


print ('\n------------Probability of MPG of Cars--------\n', Q20_data.plot(kind='density'))


# # Q21

# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis
from scipy.stats import skew
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[57]:


Q21A_data = pd.read_csv("Cars.csv")
Q21A = pd.DataFrame(Q21A_data)
print(Q21A.head())
print(Q21A.tail())


# In[58]:


sns.distplot(Q21A, label = 'MPG');
plt.xlabel('MPG');
plt.ylabel('Density');
plt.legend();


# In[59]:


Q21B_data = pd.read_csv("wc-at.csv")
Q21B = pd.DataFrame(Q21B_data)
print(Q21B.head())
print(Q21B.tail())


# In[60]:


sns.distplot(Q21B, label = 'WAIST');
plt.xlabel('Waist');
plt.ylabel('Density');
plt.legend();


# In[61]:


sns.distplot(Q21B, label = 'AT');
plt.xlabel('AT');
plt.ylabel('Density');
plt.legend();


# # Q24

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis
from scipy.stats import skew
import seaborn as sns
from scipy import stats
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


pop_mean = 270
sample_size = 18
sample_mean = 260
sample_sd = 90
standard_error = sample_sd/np.sqrt(sample_size)
t_score = (sample_mean -pop_mean)/standard_error
print('\n-------------Standard Error-------\n',standard_error)
print('\n-------------T_Score-------\n',t_score)


# In[64]:


df = sample_size - 1
alpha = 0.05
critical_value = norm.ppf(1.0 - alpha,df)
print('\n-------------Critical Value-------\n',critical_value)


# In[65]:


probability = (1-norm.cdf(abs(t_score),df))*2.0
print('\n-------------Probability Value-------\n',probability)


# In[66]:


if abs(t_score) <= critical_value:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')


# In[67]:


if probability > alpha:
	print('Accept null hypothesis that the means are equal.')
else:
	print('Reject the null hypothesis that the means are equal.')


# In[68]:


print(norm.ppf(0.05))
print(norm.rvs(size=90))


# In[ ]:




