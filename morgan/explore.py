# standard ds imports
import numpy as np
import pandas as pd

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
from wordcloud import WordCloud


# for stats
import scipy.stats as stats

def overall_data_vis(train):
    plt.figure(figsize=(12,8))
    # palette = {c: "green" if c in ['intj', 'intp', 'entj', 'entp']  else 'r' for c in df['type']}
    ax = sns.histplot(data=train,
                      x='personality_domain',
                      hue='type',
                      discrete=True, 
                      multiple='stack', 
                      palette=np.array(sns.color_palette("Blues_d", 1))
                     )

    # # iterate through each container
    # for c in ax.containers:

    #     # Optional: if the segment is small or 0, customize the labels
    #     labels = [round(v.get_height()) if v.get_height() > 0 else '' for v in c]

    #     # remove the labels parameter if it's not needed for customized labels
    #     ax.bar_label(c, labels=labels, label_type='center', fontsize=16)

    # analysts
    plt.text(-.1, 52.5, 'INTJ', fontsize=16)
    plt.text(-.1, 2, 'INTP', fontsize=16)
    plt.text(-.1, 34, 'ENTP', fontsize=16)
    plt.text(-.1, 12, 'ENTJ', fontsize=16)

    # sentinels
    plt.text(.88, 58, 'ESFJ', fontsize=16)
    plt.text(.88, 29, 'ISFJ', fontsize=16)
    plt.text(.88, 13, 'ESTJ', fontsize=16)
    plt.text(.88, 2.5, 'ISTJ', fontsize=16)

    # explorers
    plt.text(1.86, 122, 'ESTP', fontsize=16)
    plt.text(1.86, 77, 'ESFP', fontsize=16)
    plt.text(1.89, 36, 'ISTP', fontsize=16)
    plt.text(1.89, 13, 'ISFP', fontsize=16)

    # diplomats
    plt.text(2.87, 70, 'ENFJ', fontsize=16)
    plt.text(2.87, 55, 'INFP', fontsize=16)
    plt.text(2.87, 32, 'ENFP', fontsize=16)
    plt.text(2.89, 6, 'INFJ', fontsize=16)


    plt.xlabel('Personality Domain')
    plt.legend().set_visible(False)
    plt.title('Data Population Split by Personality Domain and Type')
    plt.show()    
    
    
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
    all_words = (' '.join(train.lemmatized)).split()

    all_cloud = WordCloud(background_color='white', height=1000, width=400).generate(' '.join(all_words))

    # Importing mask
    from PIL import Image

    twitter_mask = np.array(Image.open("./Twitter.png"))

    # Plot the wordcloud with the mask applied
    wc = WordCloud(background_color='skyblue', mask= twitter_mask, colormap = 'Blues',
                   contour_color='white', contour_width=1).generate(' '.join(all_words))
    plt.figure(figsize=[10,10])
    plt.tight_layout()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title('Most Common Words Overall')
    plt.show()
    
def q2a_stats(train):
    print('placeholder')
    
def q2b_vis(train):
    print('placeholder')
    
def q2b_stats(train):
    α = 0.05
    introvert = train[train.i_e == 'i'].message_length/100
    extrovert = train[train.i_e == 'e'].message_length/100

    t, pval = stats.levene(extrovert, introvert)

    t, p = stats.ttest_ind(extrovert, introvert, equal_var=(pval>α))

    if (t > 1) & (p/2 < α):
        print('''
        Reject the Null Hypothesis. 
        Findings suggest the mean tweet length is longer for extroverts than introverts.
        ''')
    else:
        print('''
        Fail to Reject the Null Hypothesis.
        Findings suggest the mean tweet length is shorter or equal for extroverts compared to introverts.
        ''')
    
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