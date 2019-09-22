import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

class PrimaryAnalysis:
    def __init__(self, df, target=None, in_cols=None):
        """
        Primary analysis class that prepares a dataset for EDA and basic modelling.

        Parameters
        ----------
        df: pandas dataframe
            Dataset to be analysed

        target: string, default None
            Name of the target column (variable to be predicted). If no target
            specified, the last column will be used.

        in_cols: array-like, default None
            List of columns to use as inputs. If not specified, all columns will
            be used (excluding the target)
        """

        sns.set(style='white', context='notebook', font_scale=1.2)

        if target is None:
            target = df.columns[-1]

        if in_cols is None:
            in_cols = list(df.columns)
            in_cols.remove(target)

        self.df = df[in_cols + [target]].copy()
        self.in_cols = in_cols.copy()
        self.target = target
        self.num_cols = []
        self.cat_cols = []
        self.ignore_cols = []

        self.clean_df()
        self.set_task_type()
        self.tt_split()

    def clean_df(self):
        """
        Clean dataset and prepare for modelling.

        Separate numerical and categorical variables.
        Remove variables that require more processing.
        Delete NAs.
        """

        for col in list(self.in_cols):
            if not self.is_num(col):
                if self.df[col].nunique() > 9:
                    self.df.drop(col, axis=1, inplace=True)
                    self.in_cols.remove(col)
                    self.ignore_cols.append(col)
                else:
                    self.cat_cols.append(col)
            else:
                self.num_cols.append(col)
        self.drop_row_count = self.df.shape[0] - self.df.dropna().shape[0]
        self.df.dropna(inplace=True)

    def is_num(self, cname):
        """
        Check if column is numerical or not.

        Columns of type 'int' with fewer than 6 unique values are considered categorical.
        """

        dt = self.df[cname].dtype
        if dt in ['float', 'int']:
            if self.df[cname].nunique() > 5:
                return True
            else:
                return False
        else:
            return False

    def set_task_type(self):
        """
        Assign model type (regression or classification) based on target variable type.

        If target variable is numerical, the task type is regression.
        If classification task, task_type is either binclass or multiclass based on number of
        levels of the target variable.
        """

        if self.is_num(self.target):
            self.num_cols.append(self.target)
            self.task_type = 'Regression'
        else:
            self.cat_cols.append(self.target)
            if self.df[self.target].nunique() == 2:
                self.task_type = 'BinClass'
            else:
                self.task_type = 'MultiClass'

    def one_hot(self):
        """
        Convert categorical variables to one hot vectors.
        """

        temp_df = self.df.copy()
        for col in self.cat_cols:
            if col != self.target:
                cats = self.df[col].unique()
                if (len(cats) != 2):
                    onehots = pd.get_dummies(temp_df[col], prefix=col, drop_first=True)
                    temp_df = pd.concat([temp_df, onehots], axis=1)
                    del temp_df[col]
                else:
                    if (0 not in cats) or (1 not in cats):
                        onehots = pd.get_dummies(temp_df[col], prefix=col, drop_first=True)
                        temp_df = pd.concat([temp_df, onehots], axis=1)
                        del temp_df[col]
        return temp_df

    def tt_split(self, one_hot=True):
        """
        Split dataset into training and test sets.
        """

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
        """
        Evaluate the results of a regression task.
        """

        rmse = np.sqrt(((preds - self.y_test) ** 2).sum() / self.y_test.shape[0])
        print('\nRMSE:',round(rmse, 3))
        plt.figure(figsize=(8,8))
        sns.scatterplot(x=self.y_test.values, y=preds)
        xy = np.linspace(self.y_test.min(), self.y_test.max(), 2)
        plt.plot(xy, xy, color='r')
        plt.title('Model Results')
        plt.xlabel('y')
        plt.ylabel('predicted values')
        plt.show()

    def class_eval(self, preds, pred_probs):
        """
        Evaluate the results of a classification task.
        """

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
        labels = self.y_test.unique()
        cm_df = pd.DataFrame(conf,
                             index = [idx for idx in labels],
                             columns = [col for col in labels])
        plt.figure(figsize=(8,6))
        cmap = sns.color_palette("Blues")
        sns.heatmap(cm_df, cmap=cmap, annot=True, fmt='')
        plt.title('Confusion Matrix')
        plt.show()

    def fit_lm(self):
        """
        Fit linear model to the dataset.
        """

        if self.task_type == 'Regression':
            self.lm = LinearRegression()
            self.lm.fit(self.x_train, self.y_train)
            preds = self.lm.predict(self.x_test)
            self.num_eval(preds)
        else:
            self.lm = LogisticRegression(solver='liblinear')
            self.lm.fit(self.x_train, self.y_train)
            preds = self.lm.predict(self.x_test)
            pred_probs = self.lm.predict_proba(self.x_test)
            self.class_eval(preds, pred_probs)

    def fit_tree(self):
        """
        Fit random forest model to the dataset.
        """

        if self.task_type == 'Regression':
            self.rf = RandomForestRegressor(n_estimators=100)
            self.rf.fit(self.x_train, self.y_train)
            preds = self.rf.predict(self.x_test)
            self.num_eval(preds)
        else:
            self.rf = RandomForestClassifier(n_estimators=100)
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
        """
        Display the correlation matrix for the dataset.
        """

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
        """
        Display KDE plots for all numeric variables in the dataset.
        """

        if len(self.num_cols) > 0:
            if len(self.num_cols) % 4 == 0:
                rows = len(self.num_cols) / 4
            else:
                rows = int(len(self.num_cols) / 4) + 1

            plt.figure(figsize=(12,2*rows))
            n = 1
            for i in self.num_cols:
                plt.subplot(rows,4,n)
                sns.kdeplot(self.df[i], legend=False)
                if len(i) > 12:
                    title = i[:12]
                else:
                    title = i
                plt.title(title)
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                n += 1
            sup_pos = 1 + (0.07 / rows)
            plt.suptitle('KDE Plots', y=sup_pos, size=14)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()

    def show_bars(self):
        """
        Display bar plots for all categorical variables in the dataset.
        """

        if len(self.cat_cols) > 0:
            if len(self.cat_cols) % 3 == 0:
                rows = len(self.cat_cols) / 3
            else:
                rows = int(len(self.cat_cols) / 3) + 1

            plt.figure(figsize=(12,3*rows))
            n = 1
            for i in self.cat_cols:
                plt.subplot(rows,3,n)
                counts = self.df[i].value_counts()
                sns.barplot([str(x) if len(str(x)) < 12 else str(x)[:12] for x in counts.index], counts.values, ci=None, alpha=0.8)
                if len(i) > 12:
                    title = i[:12]
                else:
                    title = i
                plt.xticks(rotation=30)
                plt.title(title)
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                n += 1
            sup_pos = 1 + (0.07 / rows)
            plt.suptitle('Bar Plots', y=sup_pos, size=14)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()

    def print_col_info(self):
        print('Ignored Columns:')
        for col in self.ignore_cols:
            print('\t',col)

    def print_row_info(self):
        print('Rows Removed (NAs): {}'.format(self.drop_row_count))
        print('Rows Retained: {}'.format(self.df.shape[0]))

    def print_divider(self):
        print('-'*100)

    def run_all(self):
        """
        Run all analysis on the provided dataset.

        This module will generate kde and bar plots for all variables and a correlation matrix.
        Linear and tree based models are fit to the dataset to get a rough idea of
        predictability. The random forest model is used to estimate feature importance.
        """

        self.print_divider()
        print('UNIVARIATE ANALYSIS')
        self.show_kdes()
        self.show_bars()
        self.print_divider()
        print('BIVARIATE ANALYSIS')
        self.show_corr()
        self.print_divider()
        print('LINEAR MODEL')
        self.fit_lm()
        self.print_divider()
        print('RANDOM FOREST MODEL')
        self.fit_tree()
        self.print_divider()
        self.print_col_info()
        self.print_divider()
        self.print_row_info()
