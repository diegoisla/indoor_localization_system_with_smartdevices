from utils.funciones import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pickle
import os

print("Comienza el el script train.py")
print(str(os.getcwd()))

df = read_data('data/raw/general-train-tagged.xml')

# df2 = read_data('data/raw/general-train-tagged.xml')
# df3 = read_data('data/raw/general-train-tagged.xml')
# joins...

# df_final

# Filtramos vacíos y neutros
df = df[~df['Polarity'].isin(['NONE', 'NEU'])]

# Transformamos variable target
df['Polarity'] = df['Polarity'].apply(polaridad_fun)

# Eliminamos los no españoles
df = df[df['Lang'] == 'es']

# Eliminamos los duplicados
df.drop_duplicates(subset = 'Content', inplace=True)

# Eliminamos signos puntuación
df['Content'] = df['Content'].apply(signs_tweets)

# Eliminamos links
df['Content'] = df['Content'].apply(remove_links)

# Nos cargamos stopwords
df['Content'] = df['Content'].apply(remove_stopwords)

# Aplicamos el Stemmer
df['Content'] = df['Content'].apply(spanish_stemmer)

df = df[['Content', 'Polarity']]

df.to_csv('data\\processed\\data_processed.csv')

# X_train, X_test..... split

pipeline = Pipeline(steps=[('vect',
                 CountVectorizer(max_df=0.5, max_features=1000, min_df=10,
                                 ngram_range=(1, 2))),
                ('cls', LinearSVC(C=0.2, max_iter=500))])

pipeline.fit(df['Content'], df['Polarity'])

with open('model\\selected_models\\finished_model.model', "wb") as archivo_salida:
    pickle.dump(pipeline, archivo_salida)

# predictions = pipeline.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# scores_df = scores....
# scores_df.to_csv('data\\metrics\\scores.csv')
# print(scores_df)

print('Finished train.py')