import pandas as pd 
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, BisectingKMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import scipy
from kneed import KneeLocator
from typing import Tuple, Literal, Any, Optional
import scipy.stats as stats
from gensim.models import KeyedVectors
import re
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib as mpl
from cycler import cycler

# Embeddings

def _sentence_vector(
        text: str,
        v: KeyedVectors
) -> np.ndarray:
    tokens = text.lower().split()
    vectors = [v[word] for word in tokens if word in v]

    if not vectors:
        return np.zeros(200, dtype=np.float32)

    return np.mean(vectors, axis=0)  # aggregation of each word in REASON

def _normalize(embeddings:np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings_normalized

def vectorize(
        data : np.ndarray,
        unique_reasons:np.ndarray,
        vectorizer : Literal['tf_idf', 'bio_word_vec'] = 'tf_idf',
        normalize:bool = False,
) -> Any:
    if vectorizer == 'tf_idf':
        v_ = TfidfVectorizer(ngram_range=(1,2), lowercase=False)
        embeddings = v_.fit_transform(data).toarray()
        return embeddings if not normalize else _normalize(embeddings)
    elif vectorizer == 'bio_word_vec':
        v_ = KeyedVectors.load_word2vec_format('BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True, limit=int(3e6)) 
        embeddings = np.vstack([_sentence_vector(t, v_) for t in unique_reasons])
        return embeddings if not normalize else _normalize(embeddings)
    
# Clustering models
def kmeans_model(
        n_clusters:int,
        data:np.ndarray,
        reason_count:pd.DataFrame,
        _weight_by:dict,
        weighted_by: Optional[Literal[None, 'raw_count', 'log_count', 'admission_rate', 'joint']]  = None,
        random_state:int = 1,
) -> np.ndarray:
    m = KMeans(n_clusters=n_clusters, random_state=random_state)
    
    if weighted_by is None:
        pc = m.fit(data) 
    else:
        # raw_count = reason_count.sort_index()['total_occurences_of_reason'].to_numpy()
        # log_count = np.log(1+raw_count)
        # admission_rate = (reason_count.sort_index()['percentage_of_admittance']/100).to_numpy() # admission rate per unique reason
        # joint = log_count*admission_rate
        # _weight_by = {'raw_count':raw_count, 'log_count':log_count, 'admission_rate':admission_rate, 'joint':joint, None:None}
        sample_weight=_weight_by[weighted_by]
        pc = m.fit(data, sample_weight=sample_weight)  
    
    return pc


def bkmeans_model(
        n_clusters:int,
        data:np.ndarray,
        reason_count:pd.DataFrame,
        _weight_by:dict,        
        weighted_by: Optional[Literal[None, 'raw_count', 'log_count', 'admission_rate', 'joint']]  = None,
        random_state:int = 1,
) -> np.ndarray:
    m = BisectingKMeans(n_clusters=n_clusters, random_state=random_state)
    if weighted_by is None:
        pc = m.fit(data) 
    else:
        # raw_count = reason_count.sort_index()['total_occurences_of_reason'].to_numpy()
        # log_count = np.log(1+raw_count)
        # admission_rate = (reason_count.sort_index()['percentage_of_admittance']/100).to_numpy() # admission rate per unique reason
        # joint = log_count*admission_rate
        # _weight_by = {'raw_count':raw_count, 'log_count':log_count, 'admission_rate':admission_rate, 'joint':joint, None:None}
        sample_weight=_weight_by[weighted_by]
        pc = m.fit(data, sample_weight=sample_weight) 
    
    return pc

# Optimize cluster number
def find_elbow(
        clusters_upper_bound:int,
        data: np.ndarray,
        reason_count:pd.DataFrame,
        _weight_by: dict,
        weighted_by: Optional[Literal['raw_count', 'log_count', 'admission_rate', 'joint', None]]  = None,
        random_state:int = 1,
        model_type:Literal['kmeans', 'bkmeans'] = 'kmeans',
        vectorizer:Literal['tf-idf', 'bwv'] = 'tf-idf'
) -> int:
    
    ks = range(2, clusters_upper_bound, 1)
    inertias = []
    silhouettes = []

    for k in ks:
        if model_type =='kmeans':
            # model = KMeans(n_clusters=k, random_state=random_state)
            model = kmeans_model(
                n_clusters=k,
                data=data,
                reason_count=reason_count,
                _weight_by=_weight_by,
                weighted_by=weighted_by
            )
        elif model_type =='bkmeans':
            model = bkmeans_model(
                n_clusters=k,
                data=data,
                reason_count=reason_count,
                _weight_by=_weight_by,
                weighted_by=weighted_by
            )

        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(data, model.labels_, metric='cosine'))

    kn = KneeLocator(ks, inertias, curve='convex', direction='decreasing')

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Inertia
    axes[0].plot(ks, inertias, '-o', color='blue')
    axes[0].set_title(f'Inertia  | {vectorizer} | {weighted_by} | {model_type}')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia')

    # Silhouette
    axes[1].plot(ks, silhouettes, '-o', color='green')
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Score')

    # Knee lines (only where meaningful)
    for ax in axes:
        for k in kn.all_knees:
            ax.axvline(k, linestyle='--', color='black')

    plt.tight_layout()
    plt.show()

    return kn.knee


def _remap(
        reason:str,
        model,
        data:np.ndarray,
        unique_reasons:np.ndarray
) -> int:
    '''
    Input:
        str - the reason of the encounter 
    Output:
        Tuple[int, int] - the corresponding cluster of the unweighted and weighted models respectively
    '''
    idx_of_reason = np.where(reason == unique_reasons)
    return model.predict(data[idx_of_reason])[0]

def sanity_check():
    return 'Version 2'