import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    
    
    
    
    own_recomender_defult_param = {'filter_already_liked_items':False, 
                        'filter_items':False, 
                        "recalculate_user":True}
    
    model_als_defult_param ={'factors':50, 'regularization':15, 'iterations':15, 
                             'num_threads':-1,'calculate_training_loss':False}
    
    def __init__(self, data,data_test=None,split_info=None):
        """ data - dataframe c данными
            data_test - даные для валидации, если нет и есть split_info то создаем
            split_info кортеж с инфрмацией как создать data_test (размер, поле деления) рассматривается только в слуяае отсутвя 
            data_test
        """     
        
        self.data_validation={}
        self.data_validation['status'] = False
        self.user_item_matrix = {'status':False,'matrix':None,'params':None}
        self.own_recommender_is_fit= {'status':False,'params':None}
        self.als_recommender_is_fit= {'status':False,'params':None}
        self.data = data.copy()
        self.full_data_train = data.copy() #Оставим полный объем данный , если нужно будет предсказывать по полному объему данных
        self.data_train = data.copy()
        if data_test is not None:
            self.data_test = data_test.copy()
        else:
            self.data_test = None
            if split_info:
                self.data_train,self.data_test = self.train_test_split(test_size_num = split_info[0],split_column =split_info[1])
        if  self.data_test is not None:
            self.data_validation['data'] = self.get_validation_data()
            self.data_validation['status'] = True


 
    def prefiltr_1(self,my_data,n=5000):
        df = my_data.copy()
        """Оставим только 5000 самых популярных товаров остальные переименуем в 999999"""
        popularity = my_data.groupby('item_id')['quantity'].count().reset_index()
        popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        top_5000 = popularity.sort_values('n_sold', ascending=False).head(n).item_id.tolist()
        df.loc[~df['item_id'].isin(top_5000), 'item_id'] = 999999 
        return df
    
    
    def prefiltr_2(self,data_train,n=5000):
        """Оставим только n самых популярных товаров, транзакции с остальными товрами удалим"""
        df = data_train.copy()
        popularity = df.groupby('item_id')['quantity'].count().reset_index()
        popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        top_n = popularity.sort_values('n_sold', ascending=False).head(n).item_id.tolist()
        df = df.loc[df['item_id'].isin(top_n)]  
        return df
    
    
    def prefiltr_3(self,data_train,n=5000):
        """транзакции с самыми не популярными n товрами удалим"""
        df = data_train.copy()
        not_popularity = df.groupby('item_id')['quantity'].count().reset_index()
        not_popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        not_top_n = not_popularity.sort_values('n_sold').head(n).item_id.tolist()
        df = df.loc[~df['item_id'].isin(not_top_n)]  
        return df   
    
    
    def prefiltr_4(self,data_train,weeks = 52):
        """Удалим транзакции с товарами, которые не покупали более n недель"""
        df = data_train.copy()
        old_item = df.groupby('item_id')['week_no'].max().reset_index()
        old_item = old_item.loc[old_item['week_no']>weeks,'item_id'].tolist()
        df = df.loc[df['item_id'].isin(old_item)]  
        return df
    


  
    def train_test_split(self,test_size_num,split_column):
        data_train = self.data[self.data[split_column] < self.data[split_column].max() - test_size_num]
        data_test = self.data[self.data[split_column] >= self.data[split_column].max() - test_size_num]
        return data_train, data_test
    
    
   
    def get_validation_data(self):
        result = self.data_test.groupby('user_id')['item_id'].unique().reset_index()
        result['train'] = result['user_id'].map(self.data_train.groupby('user_id')['item_id'].unique())
        result['full_train'] = result['user_id'].map(self.full_data_train.groupby('user_id')['item_id'].unique())
        result.rename(columns={'item_id':'test'},inplace=True)
        return result

 
    def prepare_matrix(self,agg_column,full=None,filtr=None):
        my_data = self.data_train.copy()
        if full:
            my_data = self.full_data_train.copy()
        if  filtr:
            for i in filtr:
                prefiltr = 'self.prefiltr_'+str(i)+'(my_data)'
                my_data = eval(prefiltr)
            
        user_item_matrix = pd.pivot_table(my_data, 
                              index='user_id', columns='item_id', 
                              values=agg_column[0], 
                              aggfunc=agg_column[1], 
                              fill_value=0
                             )
        
        user_item_matrix = user_item_matrix.astype(float) 
        self.prepare_dicts(user_item_matrix)
        self.current_working_data = my_data.copy()

        return user_item_matrix
            


    def prepare_dicts(self,user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        self.id_to_itemid = dict(zip(matrix_itemids, itemids))
        self.id_to_userid = dict(zip(matrix_userids, userids))

        self.itemid_to_id = dict(zip(itemids, matrix_itemids))
        self.userid_to_id = dict(zip(userids, matrix_userids))
        
        return  self.id_to_itemid,  self.id_to_userid,  self.itemid_to_id,  self.userid_to_id
    
    
     
    def make_data(self,agg_column,filtr=None,full =False):
        self.full = full
        uim = self.prepare_matrix(agg_column=agg_column,full=full,filtr=filtr)
        uim_w = uim.copy()
        self.user_item_matrix['uim_matrix_w'] = csr_matrix(uim_w).tocsr()
        uim[uim>0]=1
        self.user_item_matrix['uim_matrix'] = csr_matrix(uim).tocsr()
        
        self.user_item_matrix['ium_matrix_w_tfidf'] = tfidf_weight(csr_matrix(uim_w.T).tocsr())
        self.user_item_matrix['ium_matrix_tfidf'] = tfidf_weight(csr_matrix(uim.T).tocsr())
        self.user_item_matrix['ium_matrix_w_bm25'] = bm25_weight(csr_matrix(uim_w.T).tocsr())
        self.user_item_matrix['ium_matrix_bm25'] = bm25_weight(csr_matrix(uim.T).tocsr())

        self.user_item_matrix['status'] = True
        self.user_item_matrix['params'] = {'agg_column':agg_column,'filtr':filtr,'full':full}
        return self.user_item_matrix
            
        
    def precision_at_k(x, k=5):
    
        bought_list = np.array(x['test'])
        recommended_list = np.array(x['predict'])[:k]
        if len(recommended_list) == 0:
            return 0


        flags = np.isin(bought_list, recommended_list)
        precision = flags.sum() / len(recommended_list)


        return precision
        
        
    
    def fit_own_recommender(self,weighting=False):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
        
        assert self.user_item_matrix['status'], 'необходимо сначала выполнить метод make_data(self,agg_column,filtr=None,weighting=None,full =False)'
        ium = self.user_item_matrix['uim_matrix'].T
        if weighting:
            assert (weighting == 'tf_idf' or weighting == 'bm25'), 'необходимо указать weighting: tf_idf или bm25 или None'
            if  weighting == 'tf_idf':
                ium = self.user_item_matrix['ium_matrix_tfidf']
            else:
                ium = self.user_item_matrix['ium_matrix_bm25']   
        self.own_recommender = ItemItemRecommender(K=1, num_threads=-1)
        self.own_recommender.fit(ium)      
        self.own_recommender_is_fit['status'] =True
        self.own_recommender_is_fit['params'] ={'model':'ItemItemRecommender(K=1, num_threads=-1)','weighting':weighting}
        self.own_recommender_is_fit['ium']=ium
        
        return self.own_recommender
    
    
    def predict_own_recommender(self,users,N=5,params=own_recomender_defult_param):
        
        param = params.copy()
        assert self.own_recommender_is_fit['status'], 'необходимо сначала выполнить метод fit_own_recommender()'
        assert type(users) == list, 'users - должен быть списком'
        uim = self.user_item_matrix['uim_matrix']
        param['user_items'] = uim
        param['N'] = N
        answer = pd.DataFrame()
        answer['user_id']=users
        if param['filter_items']:
            param['filter_items']=[self.itemid_to_id[i] for i in params['filter_items']]
        rec=[]
        for user in users:
            param['userid'] = self.userid_to_id[user]
            rec.append( [self.id_to_itemid[i[0]] for i in self.own_recommender.recommend(**param)])
        answer['result']  = rec
        return answer

    
    
    def validation_own_recommender(self,metric=precision_at_k,N=5,params=own_recomender_defult_param):
        assert self.data_validation['status'], 'тестовые данные не созданы'
        assert self.own_recommender_is_fit['status'], 'необходимо сначала выполнить метод fit_own_recommender()'
        df = self.data_validation['data']
        users = df['user_id'].to_list()
        predict = self.predict_own_recommender(users = users,N=N,params=params)
        df['predict'] = predict['result']
        
        return df.apply(metric,axis=1).mean()
            
        
  
    def fit_als(self, params = model_als_defult_param,weighting=False):
        """Обучает ALS"""
        
        assert self.user_item_matrix['status'], 'необходимо сначала выполнить метод make_data(self,agg_column,filtr=None,weighting=None,full =False)'
        ium = self.user_item_matrix['uim_matrix_w'].T
        if weighting:
            assert (weighting == 'tf_idf' or weighting == 'bm25'), 'необходимо указать weighting: tf_idf или bm25 или None'
            if  weighting == 'tf_idf':
                ium = self.user_item_matrix['ium_matrix_w_tfidf']
            else:
                ium = self.user_item_matrix['ium_matrix_w_bm25']
        
        self.model_als = AlternatingLeastSquares(**params)
        self.model_als.fit(ium)
        self.als_recommender_is_fit['status'] = True
        self.als_recommender_is_fit['params'] = {'model':params,'weighting':weighting}
        self.als_recommender_is_fit['ium'] = ium
        
        return self.model_als
    
    
    def predict_als(self,users,N=5,params=own_recomender_defult_param):
        
        param = params.copy()
        assert self.als_recommender_is_fit['status'], 'необходимо сначала выполнить метод fit_als()'
        assert type(users) == list, 'users - должен быть списком'
        uim = self.user_item_matrix['uim_matrix_w']
        param['user_items'] = uim
        param['N'] = N
        answer = pd.DataFrame()
        answer['user_id']=users
        if param['filter_items']:
            param['filter_items']=[self.itemid_to_id[i] for i in params['filter_items']]
        rec=[]
        for user in users:
            param['userid'] = self.userid_to_id[user]
            rec.append( [self.id_to_itemid[i[0]] for i in self.model_als.recommend(**param)])
        answer['result']  = rec
        return answer
    
    
    def validation_als_recommender(self,metric=precision_at_k,N=5,params=own_recomender_defult_param):
        assert self.data_validation['status'], 'тестовые данные не созданы'
        assert self.als_recommender_is_fit['status'], 'необходимо сначала выполнить метод fit_als()'
        df = self.data_validation['data'].copy()
        users = df['user_id'].to_list()
        predict = self.predict_als(users = users,N=N,params=params)
        df['predict'] = predict['result']

        return df.apply(metric,axis=1).mean()  
    
    
    def get_recs(self,user,popularity,not_my=0):
        result = []
        for item in popularity[popularity['user_id']==user]['item_id'].to_list():
            recs_ = self.model_als.similar_items(self.itemid_to_id[item], N=3)
            recs = [self.id_to_itemid[i[0]] for i in recs_]
            if 999999 in recs:
                recs.remove(999999)
            result.append(recs[not_my])
        return  result      


    def get_similar_items_recommendation(self, users,not_my=0, N=5):
        
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров
        not_my =1 если хотим предсказать поекупку собственных товаров (вроде own_recomender), 0 - обратно"""
        assert  self.als_recommender_is_fit['status'],'Модель als не обучена, используйте fit_als()'
        assert  type(users)==list,'параметр users должен быть list'
        assert  not_my in [0,1],'параметр not_my должен быть равен 0 или 1'
        my_data = self.current_working_data.copy()
        my_data = my_data[my_data['user_id'].isin(users)]    
        popularity = my_data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)
        popularity = popularity[popularity['item_id'] != 999999]
        popularity =popularity.groupby('user_id').head(N)
        popularity.sort_values(['user_id','quantity'], ascending=False, inplace=True)
        result = pd.DataFrame()
        result['user_id'] = users
        result['similar_recommendation'] = result['user_id'].apply(\
                                            lambda x: self.get_recs(user = x,popularity = popularity,not_my=not_my))

        return result
    
    
    def validation_similar_items_recommendation(self,metric=precision_at_k,N=5,not_my=0):
        assert self.data_validation['status'], 'тестовые данные не созданы'
        assert self.als_recommender_is_fit['status'], 'необходимо сначала выполнить метод fit_als()'
        assert  not_my in [0,1],'параметр not_my должен быть равен 0 или 1'
        df = self.data_validation['data'].copy()
        users = df['user_id'].to_list()
        predict = self.get_similar_items_recommendation(users = users,N=N,not_my=not_my)
        df['predict'] = predict['similar_recommendation']

        return df.apply(metric,axis=1).mean() 
    
    
    
    def get_user(self,user):
        users = self.model_als.similar_users(self.userid_to_id[user], N=2)
        
        return  self.id_to_userid[users[1][0]]
    
    
    def get_similar_users_recommendation(self, users, N=5,params=own_recomender_defult_param):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        assert  self.als_recommender_is_fit['status'],'Модель als не обучена, используйте fit_als()'
        assert  type(users)==list,'параметр users должен быть list'
        result = pd.DataFrame()
        result['user_id'] = users
        result['simular_user_id'] = result['user_id'].apply(self.get_user)
        result['similar_recommendation'] = self.predict_als(result['simular_user_id'].to_list(),N=5,params=params)['result']

        return result    
            
    def validation_similar_users_recommendation(self,metric=precision_at_k,N=5):
        assert self.data_validation['status'], 'тестовые данные не созданы'
        assert self.als_recommender_is_fit['status'], 'необходимо сначала выполнить метод fit_als()'
        df = self.data_validation['data'].copy()
        users = df['user_id'].to_list()
        predict = self.get_similar_users_recommendation(users = users,N=N)
        df['predict'] = predict['similar_recommendation']

        return df.apply(metric,axis=1).mean()     