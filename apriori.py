#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd 
from mlxtend.preprocessing import OnehotTransactions 
from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules 
from mlxtend.preprocessing import TransactionEncoder


# In[19]:


dataset = [['Hot Dogs', 'Buns', 'Ketchup'], 
           ['Hot Dogs', 'Buns'], 
           ['Hot Dogs', 'Coke', 'Chips'], 
           ['Chips', 'Coke'], 
           ['Chips', 'Ketchup'],
           ['Hot Dogs','Coke','Chips']] 
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
df 


# In[24]:


frequent_itemsets = apriori(df, min_support=0.3334, use_colnames=True) 
print (frequent_itemsets) 

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
'''rules = association_rules(frequent_itemsets, metric-"lift", min_threshold=1.2) '''
print (rules) 
support=rules.as_matrix(columns=['support']) 
confidence=rules.as_matrix(columns=['confidence']) 
#print(support) 
#print(confidence) 
'''import random 
import matplotlib.pyplot as plt
for i in range (len(support)): 
                           support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5)
                           confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5) 
plt.scatter(support, confidence, alpha=0.5, marker="*") 
plt.xlabel('support') 
plt.ylabel('confidence') 
plt.show() '''


# In[33]:


dataset2 = [['A', 'B', 'C', 'D','E','F'], 
           ['B', 'C','D','E','F','G'], 
           ['A', 'D', 'E', 'H'], 
           ['A', 'D','F','I','J'], 
           ['B', 'D','E','K']] 
te = TransactionEncoder()
te_ary2 = te.fit(dataset2).transform(dataset2)
df2 = pd.DataFrame(te_ary2, columns=te.columns_)
df2 


# In[34]:


frequent_itemsets = apriori(df2, min_support=0.6, use_colnames=True) 
print (frequent_itemsets) 

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
'''rules = association_rules(frequent_itemsets, metric-"lift", min_threshold=1.2) '''
print (rules) 


# In[35]:


dataset3 = [['75534', '75535', '75536', '75537','75538','75539'], 
           ['75535', '75539','755374','75536','75535','75535'], 
           ['75537', '75534','75535','75536','75537','75534'], 
           ['75538', '75536','75536','755357','75537','75534'], 
           ['75534', '75536','75537','75537','75535','75539']] 
te = TransactionEncoder()
te_ary3 = te.fit(dataset3).transform(dataset3)
df3 = pd.DataFrame(te_ary3, columns=te.columns_)
df3 


# In[36]:


frequent_itemsets = apriori(df3, min_support=0.6, use_colnames=True) 
print (frequent_itemsets) 

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
'''rules = association_rules(frequent_itemsets, metric-"lift", min_threshold=1.2) '''
print (rules) 


# In[ ]:




