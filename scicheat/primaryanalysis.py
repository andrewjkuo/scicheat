import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

class PrimaryAnalysis:
    def __init__(self, df, target, in_cols=None, process=True, plot=True):

        sns.set(style='white', context='notebook', font_scale=1.2)

        if in_cols is None:
            in_cols = list(df.columns)
            in_cols.remove(target)

        self.df = df[in_cols + [target]].copy()
        self.in_cols = in_cols.copy()
        self.target = target
        self.process = process
        self.plot = plot

        self.clean_df()
        self.set_task_type()
        self.tt_split()

    def clean_df(self):
        for col in list(self.in_cols):
            if not self.is_num(col):
                if self.df[col].nunique() > 8:
                    self.df.drop(col, axis=1, inplace=True)
                    self.in_cols.remove(col)
        self.df.dropna(inplace=True)

    def is_num(self, cname):
        dt = self.df[cname].dtype
        if dt == 'float64':
            return True
        elif dt == 'int64':
            if self.df[cname].nunique() > 4:
                return True
            else:
                return False
        else:
            return False

    def set_task_type(self):
        if self.is_num(self.target):
            self.task_type = 'Regression'
        else:
            if self.df[self.target].nunique() == 2:
                self.task_type = 'BinClass'
            else:
                self.task_type = 'MultiClass'

    def one_hot(self):
        temp_df = self.df.copy()
        for col in self.in_cols:
            if not self.is_num(col):
                cats = self.df[col].unique()
                if (len(cats) != 2) or (0 not in cats) or (1 not in cats):
                    onehots = pd.get_dummies(temp_df[col], prefix=col, drop_first=True)
                    temp_df = pd.concat([temp_df, onehots], axis=1)
                    del temp_df[col]
        return temp_df

    def tt_split(self, one_hot=True):
        cutoff = int(self.df.shape[0] * 0.75)
        if one_hot:
            temp_df = self.one_hot()
        else:
            temp_df = self.df.copy()

        x_cols = [col for col in temp_df.columns if col != self.target]
        temp_df = temp_df.sample(frac=1)

        self.x_train = temp_df.iloc[:cutoff][x_cols]
        self.x_test = temp_df.iloc[cutoff:][x_cols]
        self.y_train = temp_df.iloc[:cutoff][self.target]
        self.y_test = temp_df.iloc[cutoff:][self.target]

    def num_eval(self, preds):
        rmse = np.sqrt(((preds - self.y_test) ** 2).sum() / self.y_test.shape[0])
        print('\nRMSE:',round(rmse, 3))
        if self.plot:
            plt.figure(figsize=(8,8))
            sns.scatterplot(x=self.y_test.values, y=preds)
            xy = np.linspace(self.y_test.min(), self.y_test.max(), 2)
            plt.plot(xy, xy, color='r')
            plt.title('Model Results')
            plt.xlabel('y')
            plt.ylabel('predicted values')
            plt.show()

    def class_eval(self, preds, pred_probs):
        acc = np.sum(preds == self.y_test) / self.y_test.shape[0]
        print('\nAccuracy:',round(acc, 3))
        if self.task_type == 'BinClass':
            auc = roc_auc_score(pd.get_dummies(self.y_test), pred_probs)
            print('\nAUC:', round(auc, 3))
        else:
            auc = roc_auc_score(pd.get_dummies(self.y_test), pred_probs, average='macro')
            print('\nAUC (micro):', round(auc, 3))
            auc = roc_auc_score(pd.get_dummies(self.y_test), pred_probs, average='micro')
            print('AUC (macro):', round(auc, 3))
        conf = confusion_matrix(self.y_test, preds)
        if self.plot:
            labels = self.y_test.unique()
            cm_df = pd.DataFrame(conf,
                                 index = [idx for idx in labels],
                                 columns = [col for col in labels])
            plt.figure(figsize=(8,6))
            cmap = sns.color_palette("Blues")
            sns.heatmap(cm_df, cmap=cmap, annot=True, fmt='')
            plt.title('Confusion Matrix')
            plt.show()
        else:
            print('\nConfusion Matrix:\n',conf)

    def fit_lm(self):
        if self.task_type == 'Regression':
            self.lm = LinearRegression()
            self.lm.fit(self.x_train, self.y_train)
            preds = self.lm.predict(self.x_test)
            self.num_eval(preds)
        else:
            self.lm = LogisticRegression()
            self.lm.fit(self.x_train, self.y_train)
            preds = self.lm.predict(self.x_test)
            pred_probs = self.lm.predict_proba(self.x_test)
            self.class_eval(preds, pred_probs)

    def fit_tree(self):
        if self.task_type == 'Regression':
            self.rf = RandomForestRegressor()
            self.rf.fit(self.x_train, self.y_train)
            preds = self.rf.predict(self.x_test)
            self.num_eval(preds)
        else:
            self.rf = RandomForestClassifier()
            self.rf.fit(self.x_train, self.y_train)
            preds = self.rf.predict(self.x_test)
            pred_probs = self.rf.predict_proba(self.x_test)
            self.class_eval(preds, pred_probs)
        feat_imp = pd.DataFrame(self.rf.feature_importances_,
                                           index = self.x_train.columns,
                                           columns=['importance']).sort_values('importance', ascending=False)

        rows = len(self.x_train.columns)
        plt.figure(figsize=(8,int(rows/2)+1))
        plt.title('Feature Importance')
        sns.barplot(x=feat_imp.values.flatten(),
                    y=[x if len(x) < 12 else x[:12] for x in feat_imp.index],
                    alpha=0.8, ci=None, palette='deep')
        plt.show()

    def show_corr(self):
        if self.plot:
            temp_df = self.df.copy()
            temp_df.columns = [x if len(x) < 12 else x[:12] for x in temp_df.columns]
            corr = temp_df.corr()
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            f, ax = plt.subplots(figsize=(12, 9))
            cmap = sns.diverging_palette(10, 240, n=9, as_cmap=True)
            sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
            plt.title('Correlations')
            plt.show()

    def show_kdes(self):
        num_cols = [x for x in self.df.columns if self.df[x].dtype in ['float64', 'int64']]

        if len(num_cols) % 4 == 0:
            rows = len(num_cols) / 4
        else:
            rows = int(len(num_cols) / 4) + 1

        plt.figure(figsize=(12,2*rows))
        n = 1
        for i in self.df.columns:
            if self.df[i].dtype in ['float64', 'int64']:
                plt.subplot(rows,4,n)
                sns.kdeplot(self.df[i], legend=False)
                if len(i) > 12:
                    title = i[:12]
                else:
                    title = i
                plt.title(title)
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                n += 1
        plt.suptitle('KDE Plots', size=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    def show_bars(self):
        cat_cols = [x for x in self.df.columns if self.df[x].dtype not in ['float64', 'int64']]

        if len(cat_cols) % 3 == 0:
            rows = len(cat_cols) / 3
        else:
            rows = int(len(cat_cols) / 3) + 1

        plt.figure(figsize=(12,3*rows))
        n = 1
        for i in self.df.columns:
            if self.df[i].dtype not in ['float64', 'int64']:
                plt.subplot(rows,3,n)
                counts = self.df[i].value_counts()
                sns.barplot([x if len(x) < 12 else x[:12] for x in counts.index], counts.values, ci=None, alpha=0.8)
                if len(i) > 12:
                    title = i[:12]
                else:
                    title = i
                plt.title(title)
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                n += 1
        plt.suptitle('Bar Plots', size=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    def run_all(self):
        print('-'*100)
        print('UNIVARIATE ANALYSIS')
        self.show_kdes()
        self.show_bars()
        print('-'*100)
        print('BIVARIATE ANALYSIS')
        self.show_corr()
        print('-'*100)
        print('LINEAR MODEL')
        self.fit_lm()
        print('-'*100)
        print('RANDOM FOREST MODEL')
        self.fit_tree()
