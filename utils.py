import pandas as pd
import re
import numpy as np
import os 
import datetime

class FakeRecogna2:
    def __init__(self, path_dataset):
        """Standardizes the dataset and saves it to a csv file

        Args:
            path_dataset (string): path to FakeRecogna2 dataset in a .xlsx file
        """
        df_dataset = self.importDataset(path_dataset)
        path = '.\\fakerecogna2\\fakerecogna2.csv'
        if not os.path.exists('.\\fakerecogna2'):
            os.makedirs('.\\fakerecogna2')
        df_dataset.to_csv(path, sep=',', header=True, index=False)
        
    
    def importDataset(self, path_dataset):
        if os.path.isfile('.\\fakerecogna2\\fakerecogna2.csv'):
            
            df_dataset = pd.read_csv('.\\fakerecogna2\\fakerecogna2.csv')
            return df_dataset
        
        else:            
            columns = ['id',
                    'title', 
                    'category',
                    'author',
                    'date',
                    'url',
                    'text',
                    'label',
                    'subtitle']
            
            df_dataset = pd.read_excel(path_dataset,
                                    names=columns)
            
            # standardizes dates
            df_dataset = df_dataset.loc[df_dataset['date'].dropna().index]
            df_dataset['date'] = df_dataset['date'].apply( lambda x: self.validate_date(x) )
            df_dataset['date'] = np.asarray(df_dataset['date'], dtype='datetime64')
            
            df_dataset = df_dataset.sort_values(['date'], ascending=True)

            # drops invalid dates
            df_dataset.drop(df_dataset.loc[df_dataset['date'] == '1-01-01'].index, inplace=True)
            
            df_dataset_text = df_dataset[ ['id','date','label','text'] ]
            return df_dataset_text

    
    def validate_date(self, date_text):
        if type(date_text) != str:
            date_text = str(date_text)
        date_text = date_text.replace('marÃ§o','marco')
        date_text = date_text.replace('março','marco')
        
        meses = ['janeiro',
                    'fevereiro',
                    'marco',
                    'abril',
                    'maio',
                    'junho',
                    'julho',
                    'agosto',
                    'setembro',
                    'outubro',
                    'novembro',
                    'dezembro']

        mesNumber = 0
        for i, mes in enumerate(meses):
            if mes in date_text:
                mesNumber = i+1
                
                date_text = date_text.replace(mes,str(mesNumber))
                break
        
        
        date_text = re.sub(r'\n', '', date_text)
        date_text = re.sub(r'(\| |)(\d|)\d[:|h]\d\d(:\d\d|)', '', date_text)
        date_text = re.sub(r'^.*?(Atualizada|\|)', '', date_text)
        last_edit = re.findall(r'(\d*?\/\d*?\/\d*)', date_text)
        if last_edit != []:
            date_text = last_edit[-1]
        date_text = re.sub(r'(?<=\d)( *?)( de |\.| |-)( *?)(?=\d)','/', date_text)
        date_text = re.sub(r'[^/\d]','', date_text)
        
        if date_text!='': 
            date_text = date_text.replace('/','-')
            
            day,month,year = date_text.split('-')
            year = year.split(' ')[0]
            isValidDate = True
            try:
                datetime.datetime(int(year),int(month),int(day))
            except ValueError:      
                try:
                    temp = year
                    year = day
                    day = temp
                    
                    datetime.datetime(int(year),int(month),int(day))
                    isValidDate = True
                except ValueError:
                    isValidDate = False
            
            if isValidDate:
                if len(year) == 2:
                    year = '20'+year
                date_text =  '%d-%0.2d-%0.2d' %(int(year),int(month),int(day))
            else:
                date_text = '0001-01-01'
                
        else:
            date_text = '0001-01-01'
        return date_text
    
    
def load(models=[]):
    """Loads the standardized dataset and the selected embeddings

    Args:
        models (list, optional): List of embedding models. [BERT, word2vec]

    Returns:
        dataframe, embeddings
    """
    vectors = {}
    dataset = pd.read_csv('.\\fakerecogna2\\fakerecogna2.csv', usecols=['date', 'label', 'text'], parse_dates=['date'])
    for model in models:
        vectors[model] = np.load(f'.\\fakerecogna2\\vectors\\{model}.npy', allow_pickle=True)
    if models:
        return dataset, vectors
    else:
        return dataset
    
    
def text_cleaning(text):
    """Replaces numerals, URLs and emails with standard tokens
    """
    text = re.sub(r'\S+@[.A-z0-9]+', 'EMAIL', text)
    text = re.sub(r'(?<!(\w|\-))((\d*)[%.\-,h]*\d)+', '0', text)
    text = re.sub(r'(\@|\#)[A-z0-9]+', 'HASHTAG', text)
    text = re.sub(r'''(?i)\b((?:https?:\/\/|www\d{0,3}[.]|[a-z0-9.\-\/]+[.][a-z]{2,4})(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]|))''', 'URL', text)
    return text