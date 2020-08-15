"""

@Author : Dasu Srinivas
@Description : Uses telegram api's to automate telegram tasks

"""
from telethon import TelegramClient,sync
from telethon import functions, types
from telethon.errors import SessionPasswordNeededError
import logging
import datetime
import pytz
import xlsxwriter
import pandas as pd
import argparse
import re
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from sklearn.manifold import TSNE
from gensim.models.ldamodel import LdaModel,CoherenceModel
from gensim.models import Word2Vec
from gensim.models import FastText
from wordcloud import WordCloud
from tqdm import tqdm
from sklearn.cluster import KMeans
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.types import NVARCHAR, Float, Integer, Unicode
from sqlalchemy.dialects import postgresql, mysql
from sqlalchemy import *
from collections import defaultdict

logging.basicConfig(level=logging.ERROR)


class InterActiveTelegramClient(TelegramClient):

    def __init__(self,session,api_id, api_hash, default_channel):
        '''
        @param: session,api_id,api_hash,default_channel
        @returns : None
        '''
        self.__default_channel = default_channel
        self._id_s=[]
        self.group_activity=defaultdict(dict)
        self.stop_words = list(set(stopwords.words('english')))+ list(punctuation)+['\n','----','----\n\n\n\n\n']
        self.lem =WordNetLemmatizer()

        print('Initializing InteractiveTelegramClient ...')

        super().__init__(
             session=session,api_id=api_id, api_hash=api_hash)

        print('Connecting to Telegram servers...')
        self.start()
        # Ensure you're authorized
        if not self.is_user_authorized():
            self.send_code_request(phone)
            try:
                self.sign_in(phone, input('Enter the code: '))
            except SessionPasswordNeededError:
                self.sign_in(password=input('Password: '))

    """ To get user profile """
    def profile(self):
        '''
        @param: None
        @return : Profile of the user (str)
        '''
        me = self.get_me()
        print("#########################################")
        print(" \t\tYour Profile:\t\t")
        str = " \tfirstName : {} \n" \
              " \tlastName : {} \n" \
              " \tuserName : {} \n" \
              " \tPhone : {} \n" \
              " \tId : {} \n"
        print(str.format(me.first_name, me.last_name, me.username, me.phone,me.id))
        print("#########################################")


    """ To get all the dialogs opened on user telegram account """
    def load_all_dialogs(self):
        '''
        @param: None
        @return: Returns all open dialgos of the user
        '''
        try:
            dialogs = self.get_dialogs(limit=None)

            result = "{0:20s}{1:20s}{2:20s}{3:40s}{4:20s}".format("Number", "Type", "ID", "Username", "Title") + "\n"
            for i, dialog in enumerate(dialogs, start=1):

                entity = dialog.entity
                if type(entity).__name__ == 'Channel':
                    result += "{0:20s}{1:20s}{2:20s}{3:40s}{4:20s}".format(str(i),"Channel/MetaGroup", str(entity.id), str(entity.username),
                                                                str(entity.title)) + "\n"


                elif type(entity).__name__ == "Chat":
                    result += "{0:20s}{1:20s}{2:20s}{3:40s}{4:20s}".format(str(i),"Group", str(entity.id), "-----------",
                                                                    str(entity.title)) + "\n"

                elif type(entity).__name__ == "User":
                    result += "{0:20s}{1:20s}{2:20s}{3:40s}{4:20s}".format(str(i),"User", str(entity.id), str(entity.username),
                                                                    str(entity.first_name)) + "\n"
                else:
                    result += "{0:20s}{1:20s}{2:20s}{3:40s}{4:20s}".format(str(i), "None", str(entity.id),
                                                                           "-----------",
                                                                          str(dialog.name)) + "\n"


            fHandler = open('contacts.txt', 'w', encoding='utf-8')

            fHandler.write(result)
            fHandler.close()

        except Exception as e :
            logger.elog("Telegram Client Can not Fetch All Dialogs Because : " + str(e))
            return None


    """ It will serach for url's in the text and replace them with space """
    def removing_url(self,text):
        '''
        @param: text 
        type: text object
        @return: text without url
        type: text object
        '''
        text = re.sub('(?P<url>https?://[^\s]+)',' ',text)
        return text


    """ It will extract words from the text by replacing all other with space """
    def only_words(self,text):
        '''
        @param: text 
        type: text object
        @return: text with words
        type: text object
        '''
        text = re.sub('\W+',' ',text)
        return text


    """  It will perform cleaning processing by removing stop words, punctuations and words with len less than 3 as they dont have significance in determining topic """
    def cleaning(self,text):
        '''
        @param: text 
        type: text object
        @return: text without single characters
        type: text object
        '''
        # remove all single characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

        # Remove single characters from the start
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)

        # Substituting multiple spaces with single space
        text = re.sub(r'\s+', ' ', text, flags=re.I)

        # Removing prefixed 'b'
        text = re.sub(r'^b\s+', '', text)

        text = text.lower()
        words = word_tokenize(text)
        words = [w for w in words if w not in self.stop_words and len(w) > 3]
        lemma = [self.lem.lemmatize(w,'v') for w in words]
        return lemma

    
    """ It will read data from the database """
    def read_from_database(self):
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="telegram_user",
                               pw="tele123",
                               db="telegram"),pool_pre_ping=True)
        #LIMIT 0, 200
        df = pd.read_sql("select message from telegram.tele_chat", engine);
        df.to_csv("message_data.csv",sep='\t',index=None)
        return df


    """ Applyting all the cleaning methods on the dataframe to get a clean_doc """
    def data_cleaning(self):
        df = self.read_from_database()
        df = self.remove_duplicate(df)
        df['removing url'] = df['message'].apply(self.removing_url)
        df['only words'] = df['removing url'].apply(self.only_words)
        df['clean messages'] = df['only words'].apply(self.cleaning)
        df.head()
        return list(df['clean messages'].values)
        

    """  Generating Document Term Matrix which will have all words with some assigned frequency according to their occurance in the Document topic matrix """
    def createDocTermMatrix(self):
        messages = self.data_cleaning()
        messages = [x for x in messages if x]
        dictionary = Dictionary(messages)
        corpus = [dictionary.doc2bow(doc) for doc in messages]
        return dictionary, corpus


    def tfidf_vectorizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        message_tfidf =[" ".join(msg) for msg in self.data_cleaning()]
        tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.01, use_idf=True, ngram_range=(1,3), norm='l2')
        tfidf_matrix = tfidf_vectorizer.fit_transform(message_tfidf)
        print(tfidf_matrix)
        return tfidf_vectorizer, tfidf_matrix


    def calc_terms_score(self):
        tfidf_vectorizer, tfidf_matrix = self.tfidf_vectorizer()
        terms = tfidf_vectorizer.get_feature_names()

        print("Number of terms: ",len(terms))
        ## sum tfidf frequency of each term through documents
        sums = tfidf_matrix.sum(axis=0)

        ## connecting term to its sums frequency
        data = []
        for col, term in enumerate(terms):
            data.append( (term, sums[0,col] ))

        ## You can sort words based IDF score or create a word cloud for better data visualization.
        ranking = pd.DataFrame(data, columns=['term','rank'])
        ranking.sort_values('rank', inplace=True, ascending=False)

        weights = {}
        for index, row in ranking.iterrows() :
            weights.update({row['term'] : row['rank']})

        return ranking, weights


    def create_wordcloud(self,weights):
        plt.figure(figsize=(20,10), facecolor='k')
        wc = WordCloud(width=1600, height=800)
        wc.generate_from_frequencies(weights)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()


    def createWord2Vec(self):
        corpus = self.data_cleaning()
        model_W2v = Word2Vec(sentences=corpus, size=100, window=5, min_count=15, workers=4, sg=0)
        return model_W2v


    def createFastText(self):
        corpus = self.data_cleaning()
        model_Ft = FastText(corpus, size=100, window=5, min_count=15, workers=4,sg=1)
        return model_Ft


    def tsne_plot(self,model):
        "Creates and TSNE model and plots it"
        labels = []
        tokens = []

        for word in model.wv.vocab:
            tokens.append(model[word])
            labels.append(word)
    
        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
        
        plt.figure(figsize=(24, 24)) 
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        plt.show()


    """  Creates LDA Model based on the Document term matrix and num of topics """
    def createLDAModel(self,no_of_topics):
        dictionary,corpus = self.createDocTermMatrix()
        ldamodel = LdaModel(corpus=corpus,id2word=dictionary,num_topics=no_of_topics,random_state=45,update_every=2,passes=40,chunksize=70,alpha='auto')
        print(ldamodel.print_topics())
        print(ldamodel.log_perplexity(corpus))
        return ldamodel,dictionary,corpus 

    def remove_duplicate(self,df):
        df.drop_duplicates(subset ="message", keep = False, inplace = True) 
        return df

    def createLDAMalletModel(self,no_of_topics):
        mallet_path="c://Documents/"
        dictionary,corpus = self.createDocTermMatrix()
        ldamalletmodel = gensim.models.wrappers.LdaMallet(mallet_path,corpus=corpus,id2word=dictionary,num_topics=no_of_topics)
        print(ldamalletmodel.print_topics())
        print(ldamalletmodel.log_perplexity(corpus))
        return ldamalletmodel,dictionary,corpus 

    def documentTopicMatrix(self, model, doc_term_matrix):
        import operator
        from functools import reduce
        #doc_term_matrix = [x for x in doc_term_matrix if x] 
        a = reduce(operator.concat, model[doc_term_matrix])
        d = defaultdict(list)
        for tup in a:
            d[tup[0]] += (tup[1],)
        #dict((k, v) for k, v in d.items() if v is not None)
        max_key= max(d,key= lambda x: len(d[x]))
        max_len = len(d[max_key])
        for item in d.keys():
            if len(d[item]) != max_len:
                diff = max_len- len(d[item])
                while diff>0:
                    d[item].append(0.0)
                    diff-=1
        print(len(d[0]))
        #dict((k, v) for k, v in d.items() if v)
        # Need to introduce values in one of the key to make it equal size
        df = pd.DataFrame.from_dict(d)
        doc_topic_matrix = df.values
        return doc_topic_matrix

    def compute_optimal_no_of_topics(self,limit, modelName = 'LDAMallet', start=2, step=1):
        from operator import itemgetter
        texts = self.data_cleaning()
        coherence_values = []
        for num_topics in tqdm(range(start, limit, step)):
            if (modelName == 'LDA') :
                model, dictionary,corpus = self.createLDAModel(num_topics)
            elif (modelName == 'LDAMallet'):
                model,dictionary,corpus = self.createLDAMalletModel(num_topics)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        # Show graph
        x = range(start, limit, step)
        max_num_topic = max(zip(coherence_values,list(x)),key=itemgetter(0))[1]
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

        return max_num_topic


    """ Calculate the coherence score for the LDA model """
    def getCoherence(self):
        ldamodel,dictionary,corpus = self.createLDAModel(no_of_topics)
        clean_doc = self.data_cleaning()
        coherence = CoherenceModel(ldamodel,texts= clean_doc,dictionary=dictionary,coherence='c_v')  # we can change coherence factor to u_mass also
        return coherence.get_coherence()
        

    """ Visualize the topic importance , bigger the size - most discussed topic """
    def visualize_topic_modelling(self):
        no_of_topics = self.compute_optimal_no_of_topics(20,modelName='LDA')
        ldamodel,dictionary,corpus = self.createLDAModel(no_of_topics)
        vis = pyLDAvis.gensim.prepare(ldamodel,corpus,dictionary)
        pyLDAvis.save_html(vis,"lda_6.html")


    def kMeans(self,n_cluster,doc_topic_matrix) :
        from sklearn.metrics import silhouette_score
        km = KMeans(n_clusters=n_cluster)
        pred = km.fit_predict(doc_topic_matrix)
        # Sum_of_squared_distances.append(km.inertia_)
        score = silhouette_score(doc_topic_matrix, pred, metric='euclidean')
        sum_of_squared_distance = km.inertia_

        return score, sum_of_squared_distance

    
    def OptimalK(self, num_topics=5, limit=50, modelName = 'LDAMallet', start = 2):
        from operator import itemgetter
        if (modelName == 'LDA'):
            model, dictionary, doc_term_matrix = self.createLDAModel(num_topics)
        elif (modelName == 'LDAMallet'):
            model, dictionary,doc_term_matrix = self.createLDAMalletModel(num_topics)

        doc_topic_matrix = self.documentTopicMatrix(model, doc_term_matrix)
        Sum_of_squared_distances = []
        silhouette_score_list = []
        K = range(start, limit)

        for k in tqdm(K) : 
            score, sum_of_squared_distance = self.kMeans(k,doc_topic_matrix)
            silhouette_score_list.append(score)
            Sum_of_squared_distances.append(sum_of_squared_distance)
        
        max_num_cluster = max(zip(silhouette_score_list,list(K)),key=itemgetter(0))[1]
        plt.plot(K, silhouette_score_list, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Silhouette score')
        plt.title('Silhouette_score for optimal k')
        plt.show()
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow_method for optimal k')
        plt.show()
        return max_num_cluster

    def kmeansClustering (self,num_topics, n_clusters, modelName = 'LDAMallet') :

        if (modelName == 'LDA'):
            model, dictionary, doc_term_matrix, = self.createLDAModel(num_topics)
        elif (modelName == 'LDAMallet'):
            model, dictionary,doc_term_matrix, = self.createLDAMalletModel(num_topics)
        
        doc_topic_matrix = self.documentTopicMatrix(model, doc_term_matrix)

        km = KMeans(n_clusters=n_clusters)
        km.fit(doc_topic_matrix)
        actual_messages= self.read_from_database()
        actual_messages = self.remove_duplicate(actual_messages)
        clustered_Message = self.zipClusterMessages(km, actual_messages['message'])
        return clustered_Message

    def zipClusterMessages(self,km, actual_messages):
        clustered_Message = []
        # Get cluster assignment labels
        clusters = km.labels_.tolist()
        for i in sorted(zip(clusters, map(str, actual_messages))):
            clustered_Message.append(i)
        return clustered_Message

    def saveClusteredMessagetToExcel(self,file_name, clustered_Message):
        # Create a workbook and add a worksheet.
        workbook = xlsxwriter.Workbook(file_name)
        worksheet = workbook.add_worksheet("clustered")

        # Iterate over the data and write it out row by row.
        for row, line in enumerate(clustered_Message):
            for col, cell in enumerate(line):
                worksheet.write_string(row, col, str(cell))
        workbook.close()

    def msg_tagging(self):
        df = self.read_from_database()
        df = self.remove_duplicate(df)
        tags=[]
        messages=[]
        github_urls=[]
        for msg in df['message']:
            urls = re.findall('(?P<url>https?://[^\s]+)', msg)
            if 'Please check group DP for group rulesPlease check group DP for group rules' not in msg:
                if 'utm_source' in msg or 'https://t.me/joinchat/' in msg or 'bitcoin' in msg or 'trading' in msg or 'earn money' in msg:
                    messages.append(msg)
                    tags.append("spam")
                elif 'colearninglounge' in msg:
                    messages.append(msg)
                    tags.append('community')
                elif 'linkedin.com' in msg:
                    messages.append(msg)
                    tags.append('linkedin')
                elif 'medium.com' in msg:
                    messages.append(msg)
                    tags.append('medium')
                elif ('github.com' in msg or 'gitlab.com' in msg) and urls is not None:
                    messages.append(msg)
                    tags.append('github')
                    github_urls.append(urls)
                elif 'kaggle.com' in msg:
                    messages.append(msg)
                    tags.append('kaggle')
                elif ('coursera.org' in msg or 'udemy.com' in msg) and 'utm_source' not in msg:
                    messages.append(msg)
                    tags.append('online courses')
                elif msg.endswith('?') or1 msg.startswith(('Can','How','What','how','can','what')):
                    messages.append(msg)
                    tags.append('query/question')
                elif msg.startswith(('hello','hi','Hello','Hi')):
                    messages.append(msg)
                    tags.append('request')
                elif msg.startswith(('thanks', 'thank you', 'Thanks', 'Thank you')):
                    messages.append(msg)
                    tags.append('Acknowledgment')
                else:
                    messages.append(msg)
                    tags.append('####')
        tagging = {"message":messages,"intent_name":tags}
        tag_df=pd.DataFrame(tagging)
        tag_df.to_csv('tagging_data.csv', sep=',', columns=['message','intent_name'], index=False)
        #for i in range(len(messages)):
        #    print(messages[i],"---",tags[i])


    """ To get last 1 day user activity in the group """
    def previous_day_group_info(self):
        to_date = pytz.timezone("US/Eastern").localize(datetime.datetime.now()-datetime.timedelta(days=0))
        from_date = pytz.timezone("US/Eastern").localize(datetime.datetime.now() - datetime.timedelta(days=288))

        pre_first_msg = self.get_messages(self.__default_channel, offset_date=from_date, limit=1)[0]
        first_msg = self.get_messages(self.__default_channel, min_id=pre_first_msg.id+1, limit=1,reverse=True)[0]
        last_msg = self.get_messages(self.__default_channel, offset_date=to_date, limit=1)[0]

        messages_between = self.get_messages(self.__default_channel, min_id=first_msg.id, max_id=last_msg.id) + [first_msg, last_msg]
        return messages_between


    """ To get users who joined channel/group in last 1 day """
    def last_day_joined_users(self):
        messages = self.previous_day_group_info()
        for msg in messages:
            if msg.message is None:
                self._id_s.append(getattr(msg,'from_id'))


    """  Greeting users who joined the Group  """
    def welcome_to_group(self):
        self.last_day_joined_users()
        msg =''
        for user_id in self._id_s:
            if self.get_entity(user_id).username is None:
                msg += "@"+self.get_entity(user_id).first_name+(self.get_entity(user_id).last_name if self.get_entity(user_id).last_name is not None else '') + ", "
            else:
                msg +="@"+self.get_entity(user_id).username + ", "

        msg += " Welcome to the Co-learning Lounge Telegram channel."
        self.send_message(self.get_me(), msg, parse_mode="Markdown")


    """ Query messages for a url, store url and user details"""
    def query_messages_for_url(self):
        user_id = []
        first_name = []
        last_name = []
        message = []
        url_data = []
        url_info = defaultdict(dict)
        for key in self.group_activity.keys():
            for msg in self.group_activity[key]['messages']:
                urls = re.findall('(?P<url>https?://[^\s]+)', msg)
                for url in urls:
                    if len(url)> 4:
                        if key not in url_info:
                            url_info[key]['messages']=[url]
                            url_info[key]['first_name']=self.group_activity[key]['first_name']
                            user_id.append(key)
                            first_name.append(self.group_activity[key]['first_name'])
                            last_name.append(self.group_activity[key]['last_name'])
                            message.append(msg)
                            url_data.append(url)
                        else:
                            url_info[key]['messages'].append(url)
                            user_id.append(key)
                            first_name.append(self.group_activity[key]['first_name'])
                            last_name.append(self.group_activity[key]['last_name'])
                            message.append(msg)
                            url_data.append(url)
        print(url_info)
        for i in range(len(message)):
            #message[i]= re.sub('/(?:\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2})','',message[i])
            message[i]=''.join(char for char in message[i] if len(char.encode('utf-8')) < 3)  # To remove characters which will take more than 4 bytes to store

        for i in range(len(last_name)):
            if last_name[i] is not None:
                last_name[i]=''.join(char for char in last_name[i] if len(char.encode('utf-8')) < 3)

        data = {'user_id': user_id, 'first_name': first_name, 'last_name':last_name,'message': message,'url':url_data}
        self.save_to_sql(pd.DataFrame(data),'url_info')


    """   To get previous day contriutors storing data in a dataframe """
    def previous_day_contributors(self):
        info = self.previous_day_group_info()
        messages = []
        user_name= []
        user_id = []
        datetime = []
        first_name =[]
        reply_to=[]
        last_name=[]
        message_id=[]
        for msg in info:
            if msg.message is not None:
                if msg.from_id not in self.group_activity:
                    self.group_activity[msg.from_id]['first_name']=self.get_entity(msg.from_id).first_name
                    first_name.append(self.get_entity(msg.from_id).first_name)
                    self.group_activity[msg.from_id]['last_name']=self.get_entity(msg.from_id).last_name
                    last_name.append(self.get_entity(msg.from_id).last_name)
                    self.group_activity[msg.from_id]['user_name']=self.get_entity(msg.from_id).username
                    user_name.append(self.get_entity(msg.from_id).username)
                    self.group_activity[msg.from_id]['messages']=[msg.message]
                    messages.append(msg.message)
                    self.group_activity[msg.from_id]['datetime']=[msg.date]
                    datetime.append(msg.date)
                    message_id.append(msg.id)
                    user_id.append(msg.from_id)
                    reply_to.append(msg.reply_to_msg_id)
                else:
                    self.group_activity[msg.from_id]['messages'].append(msg.message)
                    self.group_activity[msg.from_id]['datetime'].append(msg.date)
                    first_name.append(self.get_entity(msg.from_id).first_name)
                    last_name.append(self.get_entity(msg.from_id).last_name)
                    user_name.append(self.get_entity(msg.from_id).username)
                    messages.append(msg.message)
                    datetime.append(msg.date)
                    user_id.append(msg.from_id)
                    message_id.append(msg.id)
                    reply_to.append(msg.reply_to_msg_id)

        # To idnetify images in the messages list
        for i in range(len(messages)):
            messages[i]=''.join(char for char in messages[i] if len(char.encode('utf-8')) < 3)
        for i in range(len(last_name)):
            if last_name[i] is not None:
                last_name[i]=''.join(char for char in last_name[i] if len(char.encode('utf-8')) < 3)
        data ={'first_name' :first_name, 'last_name':last_name, 'user_name':user_name,'user_id':user_id, 'message':messages,'timestamp':datetime,'reply_to_msg_id':reply_to,'message_id':message_id}
        tele_df = pd.DataFrame(data)
        tele_df['date'] = [d.date() for d in tele_df['timestamp']]
        tele_df['time'] = [d.time() for d in tele_df['timestamp']]
        #tele_df.sort_values(by=['message_id'],inplace=True,ascending=False)
        df=pd.DataFrame(self.group_activity)
        df=df.transpose()
        return tele_df


    """ Mapping dataframe datatypes to mysql datatypes """
    def mapping_df_types(self,df):
        dtypedict = {}
        for i, j in zip(df.columns, df.dtypes):
            if "object" in str(j) and ("message" in str(i) or "url" in str(i)): 
                dtypedict.update({i: NVARCHAR(length=12000)})
            else:
                dtypedict.update({i: NVARCHAR(length=255)})
            if "float" in str(j):
                dtypedict.update({i: Float(precision=2, asdecimal=True)})
            if "int" in str(j):
                dtypedict.update({i: Integer()})
        return dtypedict


    """ Saving dataframe output to mysql tabel """
    def save_to_sql(self,df,table_name):
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="telegram_user",
                               pw="tele123",
                               db="telegram"),pool_pre_ping=True)
        df.to_sql(name=table_name, con = engine, if_exists = 'append', index=False,dtype=self.mapping_df_types(df))
                    



if __name__ == "__main__":
    # For windows issue __spec__ workaround 
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-id", "--api_id", dest="api_id",type=int,
            help="It's a mandatory, Need to give appication id, example id is : 4561235",required=True,default=None)
    arg_parse.add_argument("-hash", "--api_hash", dest="api_hash",type=str,
                           help="It's a mandatory argument, need to give application hash id ", required=True,default=None)
    arg_parse.add_argument("-s", "--session", dest="session",type=str,
                           help="It's a optional argument", default=None,required=False)
    arg_parse.add_argument("-dc", "--channel", dest="channel",type=str,
                           help="Please provide channel name on which you want to perform this operation", default="default_channel")
    args = arg_parse.parse_args()
    if args.api_id is not None or args.api_hash is not None:
        try:
            client = InterActiveTelegramClient(args.session,args.api_id, args.api_hash,args.channel)
            client.profile()
            print("Previous Day contributors ...")
            #client.save_to_sql(client.previous_day_contributors(),'tele_chat')
            #client.createFastText()
            #client.tsne_plot(client.createWord2Vec())
            #client.getCoherence()
            #client.calc_terms_score()
            #topics=28#client.compute_optimal_no_of_topics(50,modelName="LDA",step=2)
            #cluster=client.OptimalK(num_topics=topics,limit=50,modelName='LDA',start=10)
            #lustered_msg=client.kmeansClustering(topics,cluster,modelName='LDA')
            #client.saveClusteredMessagetToExcel("clustering_data.xlsx",clustered_msg)
            #client.visualize_topic_modelling()
            #client.query_messages_for_url()
            client.msg_tagging()
            #print("Welcome new members")
            #client.welcome_to_group()
            client.disconnect()
        except Exception as e :
            print("Can not Create Client Because : "+str(e))
            print("Please relaunch client !")
