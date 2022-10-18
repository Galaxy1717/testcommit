import numpy as np
import json
from newspaper import Article

with open('referencefrequencies.json','r') as f:
    frequencies=json.load(f)
    
# extract keywords from corpus and find external data from these keywords
class DataAugmenter:
    
    def __init__(self,nbkeywords=250):
        # number of keywords to find external data from
        self.nbkeywords=nbkeywords
        # dictionnaries of frequencies, see exemple above
        self.frequencies=frequencies
    
    # remove citations '[k]' from wikipedia articles, usually leads to problems otherwise
    def removecitations(self,string):
        firstpos=[]
        endpos=[]
        for k in range(len(string)):
            if string[k]=='[':
                firstpos.append(k)
            if string[k]==']':
                endpos.append(k)
        if len(firstpos)!=len(endpos):
            return ''
        if firstpos==[]:
            return string
        newstring=''
        lastpos=0
        for i,j in zip(firstpos,endpos):
            newstring+=string[lastpos:i]
            lastpos=j+1
        newstring+=string[lastpos:]
        return newstring
    
    # check if all the words of a gien entity belong to the english dictionnary self.dicoeng
    def check_if_english(self,entity):
        for word in entity.split():
            if word not in self.frequencies:
                return False
        return True
    
    # get average frequency of an entity
    def get_tfidf(self,dictpassages,entity):
        nbappear=0
        tf=0
        for passage in dictpassages.values():
            if passage!=[]:
                passagetf=passage.count(entity)/len(passage)
                tf+=passagetf
                #tf+=np.log(1+passage.count(entity)/len(passage))
            if passagetf>0:
                nbappear+=1
        if nbappear==0:
            return 0
        return tf*np.log(len(dictpassages)/nbappear)
        
    # input: dictionnaries of entities of passages, output: list of keywords sorted by relevance score
    def extract_keywords(self,dictpassages,vocab):
        listscores=[]
        for entity in vocab:
            score=self.get_tfidf(dictpassages,entity)
            listscores.append((entity,score))
        # sort by relevance score
        sortedentities=sorted(listscores, key=lambda tup: tup[1],reverse=True)
        # output more keywords than self.nbkeywords because some of them cannot lead to an external data
        return list(map(lambda x:x[0],sortedentities[:3*self.nbkeywords]))
            
    # extract keywords then build new paragraphs from wikipedia articles
    def augment(self,dictpassages,vocab):
        newparagraphs=[]
        keywords=self.extract_keywords(dictpassages,vocab)
        # count how many articles have been found
        nbprocessed,nbkeywords=0,self.nbkeywords
        suburl='https://en.wikipedia.org/wiki/'
        for keyword in keywords:
            # check if enough data
            if nbprocessed<nbkeywords:
                keyword='_'.join(keyword.split())
                try:
                    article=Article(suburl+keyword)
                    article.download()
                    article.parse()
                    paragraphs=article.text.split('\n\n')
                    paragraphs=list(map(self.removecitations,paragraphs))
                    newparagraphs+=paragraphs
                    nbprocessed+=1
                except:
                    continue
        return newparagraphs

with open('sifenglishdict.json') as json_file:
    weight = json.load(json_file)

# input: model: embedding model, weight: sif weights (exemple stored in sifenglishdict.json), 
# sentences: pre-processed list of sentences to embed ---> output: list of embeddings
def getsifemb(model,sentences,weight=weight):
    listemb=[]
    for s in sentences:
        size=model[model.index_to_key[0]].shape[0]
        # embedding vector of sentence s
        vs=np.zeros(size)
        s=s.split()
        # compute weighted average of the sentence
        for word in s:
            vs+=weight[word]*model[word]
        listemb.append(list(vs/len(s)))
    # compute singular values of the matrix of embeddings, then normalize the embeddings
    arremb=np.array(listemb).T
    U, S, VH = np.linalg.svd(arremb, full_matrices=True)
    u=np.array([U[:,0]])
    for k in range(len(listemb)):
        listemb[k]=arremb[:,k]-(u.T@u)@arremb[:,k]
    return listemb