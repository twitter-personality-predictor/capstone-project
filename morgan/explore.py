# standard ds imports
import numpy as np
import pandas as pd

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)



# for stats
import scipy.stats as stats

def overall_data_vis(train):
    print('placeholder for pie chart here')
    
def q1_vis(train):
    plt.figure(figsize=(12,8))
    sns.kdeplot(train[train.personality_domain == 'analyst'].sentiment, label='analyst', color='blue')
    plt.axvline(train[train.personality_domain == 'analyst'].sentiment.mean(), color='blue')
    sns.kdeplot(train[train.personality_domain == 'diplomat'].sentiment, label='diplomat', color='red')
    plt.axvline(train[train.personality_domain == 'diplomat'].sentiment.mean(), color='red')
    sns.kdeplot(train[train.personality_domain == 'explorer'].sentiment, label='explorer', color='orange')
    plt.axvline(train[train.personality_domain == 'explorer'].sentiment.mean(), color='orange')
    sns.kdeplot(train[train.personality_domain == 'sentinel'].sentiment, label='sentinel', color='green')
    plt.axvline(train[train.personality_domain == 'sentinel'].sentiment.mean(), color='green')
    plt.xlim(-1,1)
    plt.title('Explorers have a More Positive Compound Sentiment on Average')
    plt.legend()
    plt.show()
    
def q1_stats(train):
    α = 0.05
    analysts = train[train.personality_domain == 'analyst'].sentiment
    sentinel = train[train.personality_domain == 'sentinel'].sentiment

    t, pval = stats.levene(analysts, sentinel)

    t, p = stats.ttest_ind(analysts, sentinel, equal_var=(pval>α))

    if (t > 1) & (p/2 < α):
        print('''
        Reject the Null Hypothesis. 
        Findings suggest the mean sentiment for sentinels is higher than analysts.
        ''')
    else:
        print('''
        Fail to Reject the Null Hypothesis.
        Findings suggest there is less than or equal mean sentiment between analysts and sentinels.
        ''')

def q2a_vis(train):
    print('placeholder')
    
def q2a_stats(train):
    print('placeholder')
    
def q2b_vis(train):
    print('placeholder')
    
def q2b_stats(train):
    print('placeholder')
    
def q3_vis(train):
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=(20,8))
    sns.boxplot(ax=axes[0], data=train[train.personality_domain == 'analyst'], x='type', y='word_count')
    axes[0].set_ylabel('Word Count')
    axes[0].set_xlabel('Analysts')
    sns.boxplot(ax=axes[1], data=train[train.personality_domain == 'diplomat'], x='type', y='word_count')
    axes[1].set_ylabel('')
    axes[1].set_xlabel('Diplomats')
    sns.boxplot(ax=axes[2], data=train[train.personality_domain == 'explorer'], x='type', y='word_count')
    axes[2].set_ylabel('')
    axes[2].set_xlabel('Explorers')
    sns.boxplot(ax=axes[3], data=train[train.personality_domain == 'sentinel'], x='type', y='word_count')
    axes[3].set_ylabel('')
    axes[3].set_xlabel('Sentinels')

    fig.suptitle('INTJs have the Least Count of Words')
    plt.tight_layout()
    plt.show()
    
def q3_stats(train):
    print('placeholder')