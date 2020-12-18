import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from time import time
from tqdm import tqdm
from collections import Counter
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectFromModel, VarianceThreshold, SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample, class_weight
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
plt.style.use('ggplot')


def add_or_remove_stop_words(remove_list, add_list):
    '''adding or removing multiple stop words instead of repetitive code'''
    
    global STOP_WORDS
    STOP_WORDS = stopwords.words('english')
    for i in range(len(remove_list)):
        STOP_WORDS.remove(remove_list[i])
    
    for i in range(len(add_list)):
        STOP_WORDS.append(add_list[i])
    return STOP_WORDS


def chi2_test(classifier, X_train, y_train, X_test, y_test, n_features):
    '''chi2 test to select best k features in selected range'''
    ch2_result = []
    for n in n_features:
        ch2 = SelectKBest(chi2, k=n)
        x_train_chi2_selected = ch2.fit_transform(X_train, y_train)
        x_validation_chi2_selected = ch2.transform(X_test)
        clf = classifier
        clf.fit(x_train_chi2_selected, y_train)
        y_pred = clf.predict(x_validation_chi2_selected)
        score = f1_score(y_test, y_pred, average='weighted')
        ch2_result.append(score)
        print("chi2 feature selection evaluation calculated for {} features".format(n))
    return ch2_result


def perf_summary(classifier, X_train, y_train, X_test, y_test):
    '''weighted f1-score summary of the classifier'''
    t0 = time()
    model_fit = classifier.fit(X_train, y_train)
    y_pred = model_fit.predict(X_test)
    train_test_time = time() - t0
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("Weighted f1 score: {0:.2f}%".format(f1*100))
    return f1, train_test_time


def nfeature_f1_checker(vectorizer, classifier, X_train, y_train, X_test, y_test, 
                              stop_words, n_features, ngram_range=(1, 3)):
    '''
    1. run pipeline to vectorize and classify for different number of features in Tf-Idf
    2. use perf_summary formula to assess the f1 score at each number of max features
    '''
    result = []
    print(classifier)
    print("\n")
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Validation result for {} features".format(n))
        nfeature_f1,tt_time = perf_summary(checker_pipeline, X_train.preprocessed_complaint, y_train, X_test.preprocessed_complaint, y_test)
        result.append((n,nfeature_f1,tt_time))
    result_df = pd.DataFrame(result,columns=['nfeatures','validation_f1_score','train_test_time'])
    return result_df


def SelectFromModel_accuracies(classifier, X_train, y_train, X_test, y_test, n_features):
    '''
    1. run selectfrommodel to use feature importances to select max features
    2. use perf_summary formula to assess the f1 score at each number of max features
    '''
    result = []
    print(classifier)
    print("\n")
    for n in n_features:
        t0 = time()
        sel_feat = SelectFromModel(classifier, threshold=-np.inf, max_features=n) 
        sel_feat.fit(X_train, y_train)
        sel_feat.get_support()
        selected_feat= X_train.columns[(sel_feat.get_support())]
        tfidf_sm_sel = sel_feat.transform(X_train)
        tfidf_test_sel = sel_feat.transform(X_test)
        print("Validation result for {} features".format(n))
        nfeature_f1,tt_time = perf_summary(classifier, tfidf_sm_sel, y_train, tfidf_test_sel, y_test)
        result.append((n,nfeature_f1,tt_time))
        run_time = time() - t0
        print("run time: {0:.2f}s".format(run_time))
        print("-"*80)
    result_df = pd.DataFrame(result,columns=['nfeatures','validation_f1','train_test_time'])
    return result_df


def chi2_feat_sel(n_features, X_train, y_train, X_test, y_test):
    '''Use the output of ch2_test formula to create new train and test sets with the optimal number of features'''
    ch2 = SelectKBest(chi2, k=n_features)
    X_train_chi2_selected = ch2.fit_transform(X_train, y_train)
    X_test_chi2_selected = ch2.transform(X_test)
    return X_train_chi2_selected, X_test_chi2_selected


def SFM_feat_sel(n_features, X_train, y_train, X_test, y_test):
    sel_feat = SelectFromModel(classifier, threshold=-np.inf, max_features=n) 
    sel_feat.fit(X_train, y_train)
    sel_feat.get_support()
    selected_feat= X_train.columns[(sel_feat.get_support())]
    tfidf_sm_sel = sel_feat.transform(X_train)
    tfidf_test_sel = sel_feat.transform(X_test)
    return tfidf_sm_sel, tfidf_test_sel


def plot_feat_sel(df, ch2_score):
    fig = plt.figure(figsize=(8,6))
    plt.plot(df.nfeatures, df.validation_f1,label='Feature Importance Selection',color='royalblue')
    plt.plot(np.arange(1000,10000,1000), ch2_score,label='chi2 feature selection',linestyle=':', color='orangered')

    plt.title("Feture Selection: Feature Importances vs Chi2")
    plt.xlabel("Number of features")
    plt.ylabel("Validation set f1_score")
    plt.legend()
    plt.show()
    return fig


def plot_roc_PR_curves(y_test, y_pred_proba, classifier:str):
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(1,2,1)
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = roc_curve(y_test_array[:,i], y_pred_proba[:,i])
        ax1.plot(fpr, tpr, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], auc(fpr, tpr)))
        
    ax1.plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
    ax1.set(xlim=[-0.05,1.0], ylim=[0.0,1.05], 
            xlabel='False Positive Rate', 
            ylabel="True Positive Rate (Recall)", 
            title=classifier + " ROC Curve")
    ax1.legend(loc="lower right")
    ax1.grid(True)

    ## Plot precision-recall curve
    ax2 = fig.add_subplot(1,2,2)
    for i in range(len(classes)):
        precision, recall, thresholds = precision_recall_curve(y_test_array[:,i], y_pred_proba[:,i])
        ax2.plot(recall, precision, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], auc(recall, precision)))
        
    ax2.set(xlim=[0.0,1.05], ylim=[0.0,1.05], 
            xlabel='Recall', 
            ylabel="Precision", 
            title=classifier + " Precision-Recall curve")

    ax2.legend(loc="best")
    ax2.grid(True)
    plt.show()
    return fig


def classif_rep(y_test, y_pred, classifier:str):
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    print('Model Results:')
    print('\n')
    print('Weighted f1-score: {}'.format(round(f1,3)))
    print('\n')
    print('Classicication Report:')
    print(classification_report(y_test, y_pred, zero_division=0))

    matrix = confusion_matrix(y_test, y_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    fig = plt.figure(figsize=(7,5))
    sns.set(font_scale=1.4)
    fig = sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2)

    # Add labels to the plot
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(classifier + ' Confusion Matrix')
    plt.show()
    return fig
