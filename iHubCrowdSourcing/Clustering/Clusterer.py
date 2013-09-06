import sys
import BoilerPlate 
import numpy
from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance
import nltk.corpus
from nltk import decorators
import nltk.stem
 
stemmer_func = nltk.stem.LancasterStemmer.stem
stopwords = set(nltk.corpus.stopwords.words('english'))

stemmer = nltk.stem.LancasterStemmer()

@decorators.memoize
def normalize_word(word):
    return stemmer_func(word.lower())
 
def get_words(stemmer, titles):
    words = set()
    for title in job_titles:
        for word in title.split():
            words.add(stemmer.stem(word.lower()))
    return list(words)
 
@decorators.memoize
def vectorspaced(stemmer, title):
    title_components = [stemmer.stem(word.lower()) for word in title.split()]
    return numpy.array([
        word in title_components and not word in stopwords
        for word in words], numpy.short)
 
if __name__ == '__main__':
 
    filename = 'CSV/pridected_true_text_alldata.csv'
    if len(sys.argv) == 2:
        filename = sys.argv[1]
 
    
    with open(filename) as title_file:
 
        job_titles = [line.strip() for line in title_file.readlines()]
 
        words = get_words(stemmer, job_titles)
 
        # cluster = KMeansClusterer(5, euclidean_distance)
        cluster = GAAClusterer(30)
        cluster.cluster([vectorspaced(stemmer, title) for title in job_titles if title])
 
        # NOTE: This is inefficient, cluster.classify should really just be
        # called when you are classifying previously unseen examples!
        classified_examples = [
                cluster.classify(vectorspaced(stemmer, title)) for title in job_titles
            ]
 
        for cluster_id, title in sorted(zip(classified_examples, job_titles)):
            print cluster_id, title