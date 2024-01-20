import re 
import warnings
warnings.filterwarnings("error")

from pathlib import Path

import os
import string 
from unicodedata import normalize
import nltk
import gensim
from gensim.models import Word2Vec
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from gensim.models.callbacks import CallbackAny2Vec
#import pandas as pd
import numpy as np
from nltk.corpus import stopwords
 
nltk.download('punkt')
nltk.download('stopwords')

import numpy 

class TextTokenizer:
    def __init__(self):

        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()
        self.lem = WordNetLemmatizer()

    def create_corpus_csv(self, file_name=None):
        if file_name != None:
            df  = pd.read_csv(file_name)
            df["c_text2"] = df["text2"].apply(self.tokenizer)
            df["c_text1"] = df["text1"].apply(self.tokenizer)
            df_new = pd.concat([df.c_text1, df.c_text2], ignore_index=True, axis=0)
            return list(df_new)
        else:
            return None

    def tokenizer(self, text):
        result = []
        strt = ""
        pattern = '[.\n]+'
        lines = re.split(pattern, text)
        # regex for removing weird chars
        re_print = re.compile('[^%s]' % re.escape(string.printable))
        regex_punct = re.compile('[%s]' % re.escape(string.punctuation))
        for line in lines:
            # unicode chars
            
            line = normalize('NFD', line).encode('ascii', 'ignore')
            
            line = line.decode('UTF-8')
            
            # split on whitespace so we can remove weird chars and punctuation
            line = line.split()
            
            # convert to lower case
            line = [word.lower() for word in line]
            
            # remove stop words
            line = [w for w in line if not w in self.stop_words] 
            
            # remove punctuation
            line = [regex_punct.sub('', word) for word in line]
            
            # remove weird chars
            line = [re_print.sub('', w) for w in line]
            
            #remove numbers
            line = [word for word in line if word.isalpha()]
            #stemming and lemmatize 
            line = [self.lem.lemmatize(w) for w in line]
            line = [ self.ps.stem(w) for w in line]
            
            #result.append(' '.join(line))
            result = result + line

        return result

loss_list = []
loss_list.append(0)

class ModelArgs:
    dim: int = 100
    window: int = 5
    epochs: int = 5
    workers: int = 4
    min_count: int = 1
    sg: int = 0  # 0 CBOW
 
class Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        now_loss = loss - loss_list[-1]
        loss_list.append(loss)
        print('Loss after epoch {}:{}'.format(self.epoch, now_loss))
        self.epoch = self.epoch + 1

class SimilarityModel:
    
    def __init__(self, args: ModelArgs, pre_train=False, model_file=None, ):
       self.epochs = args.epochs
       self.dim = 100
       if pre_train:
            self.model=Word2Vec.load(model_file)
       else:
           self.model=Word2Vec(window=args.window, min_count=args.min_count, workers=args.workers, sg=args.sg)       
    
    def train(self, text):
        self.model.build_vocab(text, progress_per=100)
        self.model.train(text, total_examples=self.model.corpus_count, compute_loss=True, epochs=self.epochs, callbacks=[Callback()])

    def save_model(self, file_name=None):
        if file_name!= None:
            self.model.save(file_name)

    def createVector(self, text):

        uniqueWords = set()
        for w in text:
            if w in self.model.wv:
                uniqueWords.add(w)

        size = len(uniqueWords)
        vec = np.zeros((size, self.dim))
        count=0
        for w in uniqueWords:
            vec[count] = self.model.wv[w]
            count+=1

        return vec


    # Calculate the cosine similarity between the vectors

    def similarityIndex(self, text1, text2):
        try:
            vec1 = self.createVector(text1)
            vec2 = self.createVector(text2)

            dot_product = sum(v1*v2 for v1, v2 in zip(vec1 ,vec2))

            # Calculate the magnitude of each vector
            magnitude_vec1 = sum(v1*v1 for v1 in vec1)**0.5
            magnitude_vec2 = sum(v2*v2 for v2 in vec2)**0.5

            # Compute cosine similarity
            cs = dot_product / (magnitude_vec1 * magnitude_vec2)

            return np.amax(cs, axis=0)
        except Exception as e:
            print(e)
            return 0

def pipeline(tokenizer, model_args, train=False,model_name=None, data_file=None):
   
    data =[]

    sim_model = SimilarityModel(model_args, pre_train=not(train), model_file=model_name)
    if train:
        data = tokenizer.create_corpus_csv(data_file)
        sim_model.train(data)

    return sim_model

if __name__ == "__main__" :
    tokenizer = TextTokenizer()
    current_path = os.path.dirname(__file__) 
#    print(current_path)
    path = os.path.join(current_path, "../data", "text.csv")
    model_path = os.path.join(current_path, "../data", "model.sim")
    #print(path)
    model = pipeline(tokenizer, ModelArgs, train=False, model_name=model_path,data_file=path)
    model.save_model(model_path)
    text1 = "'lions blow to world cup winners british and irish lions coach clive woodward says he is unlikely to select any players not involved in next year s rbs six nations championship.  world cup winners lawrence dallaglio  neil back and martin johnson had all been thought to be in the frame for next summer s tour to new zealand.  i don t think you can ever say never   said woodward.  but i would have to have a compulsive reason to pick any player who is not available to international rugby.  dallaglio  back and johnson have all retired from international rugby over the last 12 months but continue to star for their club sides. but woodward added:  the key thing that i want to stress is that i intend to use the six nations and the players who are available to international rugby as the key benchmark.  my job  along with all the other senior representatives  is to make sure that we pick the strongest possible team.  if you are not playing international rugby then it s still a step up to test rugby. it s definitely a disadvantage.   i think it s absolutely critical and with the history of the lions we have got to take players playing for the four countries.  woodward also revealed that the race for the captaincy was still wide open.  it is an open book   he said.  there are some outstanding candidates from all four countries.  and following the all blacks  impressive displays in europe in recent weeks  including a 45-6 humiliation of france  woodward believes the three-test series in new zealand will provide the ultimate rugby challenge.  their performance in particular against france was simply awesome   said the lions coach.  certain things have been suggested about the potency of their front five  but they re a very powerful unit.  with his customary thoroughness  woodward revealed he had taken soundings from australia coach eddie jones and jake white of south africa following their tour matches in britain and ireland.  as a result  woodward stressed his lions group might not be dominated by players from england and ireland and held out hope for the struggling scots.  scotland s recent results have not been that impressive but there have been some excellent individual performances.  eddie in particular told me how tough they had made it for australia and i will take on board their opinions.  and scotland forward simon taylor looks certain to get the call  provided he recovers from knee and tendon problems.  i took lessons from 2001 in that they did make a mistake in taking lawrence dallaglio when he wasn t fit and went on the trip.  every player has to be looked at on their own merits and simon taylor is an outstanding player and i have no doubts that if he gets back to full fitness he will be on the trip.  i am told he should be back playing by march and he has plenty of time to prove his fitness for the lions - and there are other players like richard hill in the same boat."
    #text2 = "broadband challeng tv view the number of european with broadband"
    #text2 = "he exploded over the past month with the web eat into tv view habit research suggest"
    #text2 = "research suggest"
    text2= "'protect whistleblowers  tuc says the government should change the law to give more protection to employees who raise health and safety concerns about their workplaces  the tuc has said.  it said data from employment tribunals suggested 1 500  safety whistleblowers  had lost their jobs since 1999. some firms found it cheaper to sack a worker than to improve buildings or change working conditions  it said. the health and safety executive said it was trying to get workers more involved in helping to make workplaces safer. the tuc figures were drawn from unfair dismissal cases at tribunals were health and safety were the main issue.  safety representatives were often ignored when raising concerns because there was no legal duty to respond  claimed the union organisation. general secretary brendan barber said:  it shouldn t be a firing offence to object to unsafe work.  workers should not be placed in the situation where they are forced to choose between risking their job or risking their personal health and safety.  mr barber  who said the  problem is far worse than official statistics show   called for a legal system that  protects safety whistleblowers . he added that workers who are not in a union  as well as casual and migrant workers   stand little chance of redress.   rory o  neill  editor of union-backed hazards magazine  which conducted the research  said:  giving union safety reps more rights in more workplaces is the ultimate win-win.  death and injuries at work increased last year  for the second time since the turn of the century.  it would be a fatal mistake not to take full advantage of the union safety effect.  the tuc has called on the government to appoint  roving  safety reps and to increase spending on health and safety work inspections. the health and safety executive had said that it had launched an initiative to make factories and offices safer  with more worker involvement.'"

    text1 = "kajal"
    text2 ="protect"
    token1 = tokenizer.tokenizer(text1)
    token2 = tokenizer.tokenizer(text2)
    score = model.similarityIndex(token1, token2)
    print("score", score)
