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
                      x='domain',
                      hue='type',
                      discrete=True, 
                      multiple='stack', 
                      palette=np.array(sns.color_palette("Blues_d", 1))
                     )
    plt.xlabel('Personality Domain')
    plt.legend().set_visible(False)
    plt.title('Data Population Split by Personality Domain and Type')
    plt.show()    
    
    
def q1_vis(train):
    plt.figure(figsize=(12,8))
    sns.kdeplot(train[train.domain == 'analyst'].sentiment, label='analyst', color='#00FFFF')
    plt.axvline(train[train.domain == 'analyst'].sentiment.mean(), color='#00FFFF', ls='--', lw='4')
    sns.kdeplot(train[train.domain == 'diplomat'].sentiment, label='diplomat', color='#1DA1F2')
    plt.axvline(train[train.domain == 'diplomat'].sentiment.mean(), color='#1DA1F2', ls='--', lw='4')
    sns.kdeplot(train[train.domain == 'explorer'].sentiment, label='explorer', color='#0047AB')
    plt.axvline(train[train.domain == 'explorer'].sentiment.mean(), color='#0047AB', ls='--', lw='4')
    sns.kdeplot(train[train.domain == 'sentinel'].sentiment, label='sentinel', color='black')
    plt.axvline(train[train.domain == 'sentinel'].sentiment.mean(), color='black', ls='--', lw='4')
    plt.text(-.64,.65,'''Explorers have
lowest average''', fontsize=16)
    plt.text(.25,.6,'''Diplomats have
highest average''', fontsize=16)
    plt.xlim(-1,1)
    plt.title('Explorers have the Only Negative Average Compound Sentiment')
    plt.xlabel('''Sentiment''')
    plt.yticks(ticks=[0,.2,.4,.6])
    plt.grid(visible=False)
    plt.legend(loc='lower left', fontsize=18)
    plt.show()
    
def q1_stats(train):
    α = 0.05
    explorer = train[train.domain == 'explorer'].sentiment
    diplomat = train[train.domain == 'diplomat'].sentiment

    t, pval = stats.levene(explorer, diplomat)

    t, p = stats.ttest_ind(explorer, diplomat, equal_var=(pval>α))

    if (t > 1) & (p/2 < α):
        print('''
        Reject the Null Hypothesis. 
        Findings suggest the mean sentiment for diplomats is higher than explorers.
        ''')
    else:
        print('''
        Fail to Reject the Null Hypothesis.
        Findings suggest there is less than or equal mean sentiment between diplomats and explorers.
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
    introvert = train[train.i_e == 'i'].char_per_tweet/100
    extrovert = train[train.i_e == 'e'].char_per_tweet/100

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
    fig, axes = plt.subplots(1, 2, figsize=(12,8))
    fig.suptitle('Explores Receive the Most Likes and Retweets on Average')
    sns.set(font_scale=2)
    sns.boxplot(ax=axes[0],data=train, y='retweetCount', x='domain',
                order=['analyst', 'diplomat', 'explorer','sentinel'],
               palette=['#89CFF0','#1DA1F2','#0047AB','#000080'])
    axes[0].set_ylim(0,1200000)
    axes[0].set_yticks(ticks=[0,300_000,600_000,900_000,1_200_000], labels=['0', '0.3', '0.6', '0.9', '1.2'])
    axes[0].set_ylabel('Retweets (millions)')
    axes[0].tick_params(axis='x', which='major', labelsize=20)

    sns.boxplot(ax=axes[1],data=train, y='likeCount', x='domain',
                order=['analyst', 'diplomat', 'explorer','sentinel'],
                palette=['#89CFF0','#1DA1F2','#0047AB','#000080'])
    axes[1].set_ylim(0,10000000)
    axes[1].set_yticks(ticks=[0,2_000_000,4_000_000,6_000_000,8_000_000, 10_000_000], labels=['0', '2', '4', '6', '8', '10'])
    axes[1].set_ylabel('Likes (millions)')
    # axes[1].set_xticks(fontsize=20)
    axes[1].tick_params(axis='x', which='major', labelsize=20)


    plt.tight_layout()
    plt.show()
    
def q3_stats(train):
    print('placeholder')