import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
from utils import load, tests

def ceildiv(a, b):
    return -(a // -b)

class Experiments:
    def __init__(self, dataset, period='W', window=2, size=40, depth=1, N=5, seed=42):
        """Sets the parameters for the experiments

        Args:
            period (str, optional): Period for dates (datetime). Defaults to weekly periods
            window (int, optional): The number of periods for each timeframe. Defaults to 2 weeks per partition
            size (int, optional): Size of each samples
            N (int, optional): Number of iterations for each experiment
            depth (int, optional): Distance of periods. Measures the tests for n previous and next windows. Defaults to 1 (only the single previous and next window)
            seed (int, optional): Random seed
        """
        self.period = period
        self.window = window
        self.size = size
        self.depth = depth
        self.N = N
        self.rng = np.random.default_rng(seed)
        
        self.models = ['BERT', 'word2vec']
        self.detectors = ['KTS','KS','LSDD', 'CVM']
        
        self.dataset = dataset
        self.partitions = ceildiv((self.dataset.date.dt.to_period(period).max() - self.dataset.date.dt.to_period(period).min()).n, self.window)+1

    def same_distribution_subset(self, vectors):
        # verdadeiro/falso, n subsets, tamanho, vetor
        subsets = np.zeros((2, self.partitions, self.size, vectors.shape[-1]))
        
        df = self.dataset.sample(frac=1).reset_index(drop=True)
        
        fake_idx = vectors[df.loc[df['label'] == 0].sample(self.size*self.partitions).index]
        true_idx = vectors[df.loc[df['label'] == 1].sample(self.size*self.partitions).index]
        
        for i in range(self.partitions):
            # fake = vectors[df.loc[df['label'] == 0].iloc[i*size:(i+1)*size].index]
            fake = fake_idx[i*self.size:(i+1)*self.size]
            subsets[0, i] = fake
            
            # true = vectors[df.loc[df['label'] == 1].iloc[i*size:(i+1)*size].index]
            true = true_idx[i*self.size:(i+1)*self.size]
            subsets[1, i] = true
        return subsets


    def same_distribution_test(self, vectors, verbose = True):
        """Performs the tests on data selected randomly from the entire dataset

        Args:
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if verbose: print(f"{self.partitions} partitions")
        
        #verdadeiro/falso, NxN, detectores
        results = np.zeros((2, self.partitions, self.partitions, self.N, len(self.detectors)))
        
        for n in range(self.N):
            if verbose: print(f"{n+1} of {self.N}:")
            subsets = self.same_distribution_subset(vectors)
            for i in range(self.partitions-1):
                for o in range(i+1, self.partitions):
                    #fake
                    results[0][i][o][n] = tests(subsets[0][i], subsets[0][o])
                    results[0][o][i][n] = results[0][i][o][n]
                    #true
                    results[1][i][o][n] = tests(subsets[1][i], subsets[1][o])
                    results[1][o][i][n] = results[1][i][o][n]
        return results

    def different_distribution(self, vectors, verbose = True):
        dates = self.dataset.date.dt.to_period(self.period)    
        period_size = ceildiv((self.dataset.date.dt.to_period(self.period).max() - self.dataset.date.dt.to_period(self.period).min()).n, self.window)+1
        if verbose: print(f"{period_size} periods")
        
        #verdadeiro/falso, window x window, repetições, detectores
        results = np.zeros((2, period_size, period_size, self.N, len(self.detectors)))
        
        for n in range(self.N):
            if verbose: print(f"{n+1} of {self.N}:")
            pstart = dates.min()
            pend = pstart + self.window
            
            i = 0
            while pend < dates.max()+2-self.depth:
                dt_window = self.dataset.loc[(self.dataset.date >= pstart.to_timestamp()) & (self.dataset.date < pend.to_timestamp())]
                true_x = self.rng.choice(dt_window.loc[dt_window.label == 1].index, self.size, replace=False)
                fake_x = self.rng.choice(dt_window.loc[dt_window.label == 0].index, self.size, replace=False)
                
                p2start = pend
                p2end = p2start + self.window
                
                o = i+1
                while p2end < pend+self.window+(self.window*self.depth):
                    dt_window = self.dataset.loc[(self.dataset.date >= p2start.to_timestamp()) & (self.dataset.date < p2end.to_timestamp())]
                    true_y = self.rng.choice(dt_window.loc[dt_window.label == 1].index, self.size, replace=False)
                    fake_y = self.rng.choice(dt_window.loc[dt_window.label == 0].index, self.size, replace=False)
                    
                    # print(pstart,pend,p2start,p2end)
                    #fake
                    results[0][i][o][n] = tests(vectors[fake_x], vectors[fake_y])
                    results[0][o][i][n] = results[0][i][o][n]
                    
                    #true
                    results[1][i][o][n] = tests(vectors[true_x], vectors[true_y])
                    results[1][o][i][n] = results[1][i][o][n]
                    p2start = p2end
                    p2end = p2start + self.window
                    o += 1
                
                pstart = pend
                pend = pstart + self.window
                i += 1
        
        return results


    def WMD(self, verbose = True):
        dates = self.dataset.date.dt.to_period(self.period)    
        period_size = ceildiv((self.dataset.date.dt.to_period(self.period).max() - self.dataset.date.dt.to_period(self.period).min()).n, self.window)+1
        if verbose: print(f"{period_size} periods")
        
        w2v = Word2Vec.load(os.getcwd()+"\\models\\tuned_bow.txt")
        
        #verdadeiro/falso, window x window, repetições, detectores
        results = np.zeros((2, period_size, period_size, self.size, self.size, self.N))
        
        for n in range(self.N):
            if verbose: print(f"{n+1} of {self.N}:")
            pstart = dates.min()
            pend = pstart + self.window
                    
            i = 0
            while pend < dates.max()+2-self.depth:
                dt_window = self.dataset.loc[(self.dataset.date >= pstart.to_timestamp()) & (self.dataset.date < pend.to_timestamp())]
                fake_x = dt_window.loc[dt_window.label == 0]['text'].sample(n=self.size, replace=False).reset_index(drop=True)
                true_x = dt_window.loc[dt_window.label == 1]['text'].sample(n=self.size, replace=False).reset_index(drop=True)
                
                p2start = pend
                p2end = p2start + self.window
                
                o = i+1
                while p2end < pend+self.window+(self.window*self.depth):
                    dt_window = self.dataset.loc[(self.dataset.date >= p2start.to_timestamp()) & (self.dataset.date < p2end.to_timestamp())]
                    fake_y = dt_window.loc[dt_window.label == 0]['text'].sample(n=self.size, replace=False).reset_index(drop=True)
                    true_y = dt_window.loc[dt_window.label == 1]['text'].sample(n=self.size, replace=False).reset_index(drop=True)
                    
                    
                    distance_matrix = np.zeros((2,self.size,self.size))
                    for j1 in range(self.size):
                        for j2 in range(j1+1,self.size):
                            distance_matrix[0][j1][j2] = w2v.wv.wmdistance(fake_x.iloc[j1],fake_y.iloc[j2])
                            distance_matrix[0][j2][j1] = distance_matrix[0][j1][j2]
                            distance_matrix[1][j1][j2] = w2v.wv.wmdistance(true_x.iloc[j1],true_y.iloc[j2])
                            distance_matrix[1][j2][j1] = distance_matrix[1][j1][j2]
                            
                    #fake
                    results[0][i][o][:,:,n] = distance_matrix[0]
                    results[0][o][i][:,:,n] = results[0][i][o][:,:,n]
                    
                    #true
                    results[1][i][o][:,:,n] = distance_matrix[1]
                    results[1][o][i][:,:,n] = results[1][i][o][:,:,n]
                    
                    p2start = p2end
                    p2end = p2start + self.window
                    o += 1
                
                pstart = pend
                pend = pstart + self.window
                i += 1
        
        return results


    def same_distribution_WMD(self, verbose=True):
        if verbose: print(f"{self.partitions} partitions")
        
        #verdadeiro/falso, NxN, detectores
        results = np.zeros((2, self.partitions, self.partitions, self.size, self.size, self.N))
        w2v = Word2Vec.load(os.getcwd()+"\\models\\tuned_bow.txt")
        
        for n in range(self.N): 
            if verbose: print(f"{n+1} of {self.N}:")
            subsets = [[pd.DataFrame()]*self.partitions,
                    [pd.DataFrame()]*self.partitions]
            
            df = self.dataset.sample(frac=1).reset_index(drop=True)
            for i in range(self.partitions):
                fake = df.loc[df['label'] == 0].iloc[i*self.size:(i+1)*self.size, 2].reset_index(drop=True)
                subsets[0][i] = fake.copy()
                
                true = df.loc[df['label'] == 1].iloc[i*self.size:(i+1)*self.size, 2].reset_index(drop=True)
                subsets[1][i] = true.copy()
                
            for i in range(self.partitions-1):
                for o in range(i+1, self.partitions):
                        
                    distance_matrix = np.zeros((2,self.size,self.size))
                    for j1 in range(self.size):
                        for j2 in range(j1+1, self.size):
                            distance_matrix[0][j1][j2] = w2v.wv.wmdistance(subsets[0][i].iloc[j1],subsets[0][o].iloc[j2])
                            distance_matrix[0][j2][j1] = distance_matrix[0][j1][j2]
                            distance_matrix[1][j1][j2] = w2v.wv.wmdistance(subsets[1][i].iloc[j1],subsets[1][o].iloc[j2])
                            distance_matrix[1][j2][j1] = distance_matrix[1][j1][j2]
                    #fake
                    results[0][i][o][:,:,n]= distance_matrix[0]
                    results[0][o][i][:,:,n] = results[0][i][o][:,:,n]
                    
                    #true
                    results[1][i][o][:,:,n] = distance_matrix[1]
                    results[1][o][i][:,:,n] = results[1][i][o][:,:,n]
        return results