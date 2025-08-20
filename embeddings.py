import utils
from sentence_transformers import SentenceTransformer
import spacy
import numpy as np
import os

class Doc2Vec:
    def __init__(self, modelName='neuralmind/bert-large-portuguese-cased'):
        self.modelName = modelName
        
    def getVector(self, dataset_train, dataset_test = None):
        model = SentenceTransformer(self.modelName)
        
        print("Generating train embeddings (SentenceTransformer)")
        x_train = model.encode(dataset_train)
        print("done")
        
        if dataset_test is not None:
            print("Generating test embeddings (SentenceTransformer)")
            x_test = model.encode(dataset_test)
            print("done")
                    
        if dataset_test is not None:
            return x_train, x_test
        else:
            return x_train
        
class Word2Vec:
    def __init__(self):
        if os.path.isfile('.\\models\\spacy.word2vec.model\\'):
            self.nlp = spacy.load('.\\models\\spacy.word2vec.model\\')
        else:
            self.nlp = self.train_model()
        
    def train_model(self):
        dataset = utils.load()
        nlp = spacy.load('pt_core_news_lg')
        docs = []
        for doc in nlp.pipe(dataset['text'].apply(utils.text_cleaning)):
            docs.append([str(token) for token in doc if token.is_alpha])
            
        model = Word2Vec(vector_size=100, min_count=3, sg=0)
        model.build_vocab(docs)

        model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)
        model.train(docs, total_examples=model.corpus_count, epochs=50)
        
        model.wv.save_word2vec_format('.\\models\\tuned_word2vec.txt')
        
        raise Exception('models\\tuned_word2vec.txt needs to be gzipped before continuing.\nThen run python -m spacy init vectors pt ./models/spacy.word2vec.model ./models/tuned_word2vec.txt.gz')
        
        
    def getVector(self, dataset, document_mean = True, text_cleaning=True):
        
        if text_cleaning:
            dataset = dataset.apply(utils.text_cleaning)
        
        embedding = []
        func = lambda x: [token.vector for token in x]
        if document_mean: 
            func = lambda x: x.vector
        for doc in self.nlp.pipe(dataset):
            embedding.append(func(doc))
        return embedding
