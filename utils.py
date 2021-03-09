
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from src.metrics import precision_at_k, recall_at_k


def prefilter_items_top_n_999999(data_train,n=5000):
    """Оставим только n самых популярных товаров, остальные переименуем в 999999"""
    df = data_train.copy()
    popularity = df.groupby('item_id')['quantity'].count().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_n = popularity.sort_values('n_sold', ascending=False).head(n).item_id.tolist()
    df.loc[~df['item_id'].isin(top_n), 'item_id'] = 999999 
    return df
    
    
def prefilter_items_top_n_del(data_train,n=5000):
    """Оставим только n самых популярных товаров, транзакции с остальными товрами удалим"""
    df = data_train.copy()
    popularity = df.groupby('item_id')['quantity'].count().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_n = popularity.sort_values('n_sold', ascending=False).head(n).item_id.tolist()
    df = df.loc[df['item_id'].isin(top_n)]  
    return df
    
    
def prefilter_items_not_top_n_del(data_train,n=5000):
    """транзакции с самыми не популярными n товрами удалим"""
    df = data_train.copy()
    not_popularity = df.groupby('item_id')['quantity'].count().reset_index()
    not_popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    not_top_n = not_popularity.sort_values('n_sold').head(n).item_id.tolist()
    df = df.loc[~df['item_id'].isin(not_top_n)]  
    return df    
 

def prefilter_items_old_weeks_n_del(data_train,weeks = 52):
    """Удалим транзакции с товарами, которые не покупали более n недель"""
    df = data_train.copy()
    old_item = df.groupby('item_id')['week_no'].max().reset_index()
    old_item = old_item.loc[old_item['week_no']>weeks,'item_id'].tolist()
    df = df.loc[df['item_id'].isin(old_item)]  
    return df



def train_test_split(data,test_size_num,split_column):
    data_train = data[data[split_column] < data[split_column].max() - test_size_num]
    data_test = data[data[split_column] >= data[split_column].max() - test_size_num]
    return data_train, data_test


def get_similar_items_recommendation(matrix_dict,df,users,model,not_my=0, N=5):

    """Рекомендуем товары, похожие на топ-N купленных юзером товаров
            not_my =1 если хотим предсказать поекупку собственных товаров (вроде own_recomender), 0 - обратно
    """

    assert  type(users)==list,'параметр users должен быть list'
    assert  not_my in [0,1],'параметр not_my должен быть равен 0 или 1'
    
    def get_recs(model,user,popularity,not_my=0):
        result = []
        for item in popularity[popularity['user_id']==user]['item_id'].to_list():
            recs_ = model.similar_items(matrix_dict['itemid_to_id'][item], N=3)
            recs = [matrix_dict['id_to_itemid'][i[0]] for i in recs_]
            if 999999 in recs:
                recs.remove(999999)
            result.append(recs[not_my])
        return  result



    my_data = df.copy()     
    my_data = my_data[my_data['user_id'].isin(users)]    
    popularity = my_data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
    popularity.sort_values('quantity', ascending=False, inplace=True)
    popularity = popularity[popularity['item_id'] != 999999]
    popularity =popularity.groupby('user_id').head(N)
    popularity.sort_values(['user_id','quantity'], ascending=False, inplace=True)
    result = pd.DataFrame()
    result['user_id'] = users
    result['similar_recommendation'] = result['user_id'].apply(\
                            lambda x: get_recs(model=model,user=x,popularity = popularity,not_my=not_my))

    return result

def get_similar_users_recommendation(matrix_dict,model,uim,users, params={'filter_already_liked_items':False, 
                        'filter_items':[999999], 
                        "recalculate_user":True}, N=5):
    """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
    assert  type(users)==list,'параметр users должен быть list'
    
    def get_user(user):
        users = model.similar_users(matrix_dict['userid_to_id'][user], N=2)
        
        return  matrix_dict['id_to_userid'][users[1][0]]
    
    
    def predict_als(users,N=5):
        
        param = params.copy()
        assert type(users) == list, 'users - должен быть списком'
        param['user_items'] = uim
        param['N'] = N
        answer = pd.DataFrame()
        answer['user_id']=users
        if param['filter_items']:
            param['filter_items']=[matrix_dict['itemid_to_id'][i] for i in params['filter_items']]
        rec=[]
        for user in users:
            param['userid'] = matrix_dict['userid_to_id'][user]
            rec.append( [matrix_dict['id_to_itemid'][i[0]] for i in model.recommend(**param)])
        answer['result']  = rec
        return answer

    
    result = pd.DataFrame()
    result['user_id'] = users
    result['simular_user_id'] = result['user_id'].apply(get_user)
    result['similar_recommendation'] = predict_als(result['simular_user_id'].to_list(),N=5)['result']

    return result

def postfilter_items():
    pass