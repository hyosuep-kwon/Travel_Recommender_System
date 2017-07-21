# -*- coding: utf-8 -*-
"""
Created on 7/21/17
Author: Jihoon Kim
"""


from konlpy.tag import Twitter
from konlpy.utils import pprint
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family="NanumBarunGothic")
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
from collections import Counter
import numpy as np
from PIL import Image
data = pd.read_csv('./data/c_data/덕수궁.csv')

text_sample = data.review

length = data.shape[0]
twit = Twitter()
pos_counter = Counter()
for i in range(length):

    twit_pos = twit.pos(text_sample[i])
    pos_counter += Counter(twit_pos)
    print("{}th Document is being processed.".format(i))

count_df = pd.DataFrame.from_dict(pos_counter, orient='index').reset_index()
count_df.rename(columns={'index': 'pos', 0: 'count'}, inplace=True)
sorted = count_df.sort_values(by='count', ascending=False)
sorted['pos_word'] = sorted.pos.apply(lambda x: x[0])
sorted['pos'] = sorted.pos.apply(lambda x: x[1])

av_pos = ['Noun', 'Verb', 'Adjective', 'Adverb']
go_index = sorted.pos.isin(av_pos)
sorted = sorted[go_index]


word = list(sorted.pos_word)
word = ' '.join(word)
palace_coloring = np.array(Image.open("./cablecar.png"))
wc = WordCloud(background_color="white", max_words=2000, mask=palace_coloring,
               max_font_size=40, random_state=42)
# generate word cloud
wc.generate(word)

# create coloring from image
image_colors = ImageColorGenerator(palace_coloring)
# recolor wordcloud and show
# we could also give color_func=image_colors directly in the constructor
plt.figure()
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

plt.figure()
plt.imshow(palace_coloring, cmap=plt.cm.gray, interpolation="bilinear")
plt.axis("off")
plt.show()