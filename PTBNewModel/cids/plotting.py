"""Wrappers on seaborn/matplotlib/plotly functions"""
# Authors: 
#     Srivatsan Ramanujam <srivatsan.ramanujam@salesforce.com>
#     Chaitanya Deepak Kondapaturi <ckondapaturi@salesforce.com>
# License: All rights owned by Customer Intelligence, Salesforce.com

#from .utils import *
#from .influence import *
from sklearn import preprocessing, metrics
import seaborn as sns, matplotlib.pyplot as plt
from operator import itemgetter
from collections import defaultdict
from textwrap import wrap
from IPython.display import display, HTML, Markdown

def distribution_plot(
        series, 
        title, 
        xlabel, 
        cumulative = False, 
        kde = False
    ):
    """Plot a histogram and cumulative distribution (if specified) of a series

    Parameters
    -----------
    series : collection (list like)
        An instance of the series to be distplotted
    title : string
        Title for the plot
    xlabel : string 
        The label for the series
    cumulative : boolean, default: ``False``
        Whether or not to also show cumulative distribution plot
    kde : boolean, default: ``False`` 
        Whether to show kernel density

    Returns
    ---------
    Plots the histogram and cumulative distribution (if specified)
        
    """
    sns.distplot(a = series, kde = kde, hist_kws = {'range': [min(series), max(series)]})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show() 
    if(cumulative):
        series_cum = sorted([(e, 1) for e in series], key = itemgetter(0))
        cum_sum_total = sum([c for e, c in series_cum])
        cum_sum_idx = defaultdict(float)
        current_sum = 0.0
        # Compute cumulative count%
        for i in range(len(series_cum)):
            current_sum += series_cum[i][1]
            cum_sum_idx[i] = current_sum
        series_cum_sum_pct = [(series_cum[i][0], cum_sum_idx[i]*100.0/cum_sum_total) for i in range(len(series_cum))]
        x, y = zip(*series_cum_sum_pct)
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel('Cumulative %')
        plt.title(title+': cumulative')
        plt.show()

def plot_prediction_error_dist(
        errors, 
        title, 
        xlabel, 
        cumulative = False, 
        kde = True
    ):
    """Show a density plot of the error in actual vs. predicted values

    Parameters
    -----------
    errors : list like
        A collection containing the errors in prediction (actual - predicted)

    title : string
        Title for the plot

    xlabel : string
        Label for the x-axis

    cumulative : boolean, default: False
        Whether to plot the cumulative distribution as well

    kde : boolean, default: True    
        Whether to compute and plot the kernel density estimate as well

    Returns
    --------
    Plots the histogram and cumulative distribution plot (if specified)

    """
    distribution_plot(errors, title, xlabel, cumulative, kde)

def plot_auc(fpr, tpr, thresholds, auc, title):
    """Plot the AUC Curve

    Parameters
    -----------
    fpr : list like
        False positive rates at different thresholds
    tpr : list like
        True positive rates at different thresholds
    thresholds : list like
        Different thresholds values corresponding to fpr and tpr
    auc : string
        AUC (Are Under the Curve) value to plot 
    title : string
        Title of the plot

    Returns
    ----------
    ROC curve plot
        
    """
    def getIndex(arr, el):
        """Return index of an element in an array/list

        Parameters
        -----------
        arr : list like
            An array/list of floating point values
        el : float
            An element in the array ``arr``

        Returns
        --------
        i : int
            Index of the element in arr that equals el

        """
        for i, v in enumerate(arr):
            if(v == el):
                return i
        return None
    
    plt.plot(fpr, tpr, color = 'salmon', label = 'ROC curve (area = %0.2f)' % auc)
    labels = ['thresh:{0:0.2f}'.format(t) for t in thresholds]
    #threshold closest to 0.4
    t_op4 = min(thresholds, key = lambda e: abs(e - 0.40))
    #threshold closest to 0.5
    t_op5 = min(thresholds, key = lambda e: abs(e - 0.50))
    #threshold closest to 0.6
    t_op6 = min(thresholds, key = lambda e: abs(e - 0.60))
    #fpr, tpr values closes to the thresholds
    t_op4_fpr, t_op4_tpr = fpr[getIndex(thresholds, t_op4)], tpr[getIndex(thresholds, t_op4)]
    t_op5_fpr, t_op5_tpr = fpr[getIndex(thresholds, t_op5)], tpr[getIndex(thresholds, t_op5)]
    t_op6_fpr, t_op6_tpr = fpr[getIndex(thresholds, t_op6)], tpr[getIndex(thresholds, t_op6)]
    for label, x, y in zip(labels, fpr, tpr):
        if(x in (t_op4_fpr, t_op5_fpr, t_op6_fpr) and y in (t_op4_tpr, t_op5_tpr, t_op6_tpr)):
            # annotate points on Matplotlib plots
            # http://stackoverflow.com/questions/5147112/
            # matplotlib-how-to-put-individual-tags-for-a-scatter-plot
            plt.annotate(
                label,
                xy = (x, y), 
                xytext = (100, 0),
                textcoords = 'offset points', 
                ha = 'right', 
                va = 'bottom',
                bbox = dict(boxstyle = 'round, pad = 0.5', fc='yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3, rad = 0')
            )
    plt.plot([0, 1], [0, 1], color = 'navy', linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc = "lower right")
    plt.show()
    
def plot_precision_recall_curve(prec, rec, thresholds, title):
    """Plot the Precision-Recall Curve

    Parameters
    -----------
    prec : list like
        Precision values at different thresholds
    rec : list like
        Recall values at different thresholds
    thresholds : list like
        Different thresholds values
    title : string
        Title of the plot

    Returns
    --------
    Plots the Precision-Recall Curve

    """
    def getIndex(arr, el):
        """Return index of an element in an array/list

        Parameters
        -----------
        arr : list like
            An array/list of floating point values
        el : float
            An element in the array ``arr``

        Returns
        --------
        i : int
            Index of the element in arr that equals el

        """
        for i, v in enumerate(arr):
            if(v == el):
                return i
        return None
    plt.plot(rec, prec, color='salmon')
    threshold_ranges = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_ranges_closest = {}
    labels = ['thresh:{0:0.2f}'.format(t) for t in thresholds]
    for tr in threshold_ranges:
        tclosest = min(thresholds, key = lambda e: abs(e-tr))
        threshold_ranges_closest[tr] = {
                                            # threshold closest to specific value in range
                                            'closest': tclosest,
                                            # recall at tclosest
                                            'recall': rec[getIndex(thresholds, tclosest)],
                                            # precision at tclosest
                                            'precision': prec[getIndex(thresholds, tclosest)]
                                       }
    x_valid_rec = set([v['recall'] for k,v in threshold_ranges_closest.items()])
    y_valid_prec = set([v['precision'] for k,v in threshold_ranges_closest.items()])
    for label, x, y in zip(labels, rec, prec):
        if(x in x_valid_rec and y in y_valid_prec):
            # annotate points on Matplotlib plots
            # http://stackoverflow.com/questions/5147112/\
            # matplotlib-how-to-put-individual-tags-for-a-scatter-plot
            plt.annotate(
                label,
                xy = (x, y), 
                xytext = (100, 0),
                textcoords = 'offset points', 
                ha = 'right', 
                va = 'bottom',
                bbox = dict(boxstyle = 'round, pad = 0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3, rad = 0')
            )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc = "lower right")
    plt.show()
    
def calculate_metrics(ytest, yscore):
    """Calculate AUC and Plot ROC curve

    Get Precision - Recall list for all the thresholds  and get the best threshold based on maximum f1 score. 
    This is used in the evaluation of ordinal binary XGBoost classification model.

    Parameters
    -----------
    ytest : list like
        Actual binary outcomes for all the test records
    yscore : list like
        Predicted binary outcomes for all the test records from XGBoost binary classifier

    Returns
    --------
    thrsh : float
        Threshold for which f1 score is maximum
    inx : int
        Index in the ytest whose probability values are less than the threshold value in "thrsh"
    
    """
    sns.set(style = 'darkgrid')
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    fpr, tpr, thresholds = metrics.roc_curve(ytest, yscore[:, 1], pos_label = 1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr,
             tpr,
             linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw = 2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve (area = {0:0.2f})'.format(roc_auc))
    plt.show()
    prec = dict()
    rec = dict()
    thresholds_prec_recall = dict()
    f1score = dict()
    prec_recall_thresh_lst =[]
    prec, rec, thresholds_prec_recall = metrics.precision_recall_curve(ytest, yscore[:, 1])
    f1score = np.array([2.0*p*r/(p+r) for p, r in zip(prec, rec)])            
    prec_recall_thresh_lst.extend(
            [(t, p, r, f) 
                    for t, p, r, f in zip(
                            thresholds_prec_recall, 
                            prec, 
                            rec, 
                            f1score
                        )
            ]
        )
    prec_recall_thresh_lst_df = pd.DataFrame(
                                    prec_recall_thresh_lst, 
                                    columns = [
                                        'threshold',
                                        'precision',
                                        'recall',
                                        'f1score'
                                    ]
                                )
    thrsh = prec_recall_thresh_lst_df.loc[prec_recall_thresh_lst_df['f1score'].idxmax()]
    inx = np.argwhere(yscore[:,1] < thrsh.threshold)
    return thrsh, inx

def plot_train_test_curve(eval_results, eval_metric, title, xlabel, ylabel):
    """Plot the train vs. test error

    Parameters
    -----------
    eval_results :  xgb_classifier.evals_result()
        Evaluation results from the XGBClassifier instance
    eval_metric : string
        Classifier evaluation metric or the cost function
    title : string
        Title of the plot 
    xlabel : string
        Label for x-axis for the plot
    ylabel : string
        Label for y-axis of the plot

    Returns
    --------
    Plot of (train vs. test error(eval_metric)) vs. No. of estimators
        
    """
    epochs = len(eval_results['validation_0'][eval_metric])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, eval_results['validation_0'][eval_metric], label = 'Train')
    ax.plot(x_axis, eval_results['validation_1'][eval_metric], label = 'Test')
    ax.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc = "top right")
    plt.show()
    
def plot_feature_importance_scores(x, y, order, title, xlabel, ylabel):
    """Draw a barplot to depict feature importance scores

    Parameters
    -----------
    x : list like
        List of ordered influence parameter in predicting the outcome
    y : list like
        List of features in the order of their importance
    order : list like
        Order of features for plotting
    title : string
        Title of the plot 
    xlabel : string
        Xlabel for the plot
    ylabel : string
        Xlabel for the plot

    Returns
    --------
    Plot of feature importance
        
    """
    plt.figure(figsize = (8, 8))
    sns.set(font_scale = 0.80)
    sns.barplot(
            x = np.ravel(list(x)), 
            y = np.ravel(list(y)), 
            orient = 'h', 
            order = np.ravel(list(order)), 
            color = 'blue'
        )
    plt.xlabel(xlabel, size = 16)
    plt.ylabel(ylabel, size = 16)
    plt.title(title, size = 16)
    plt.show()
    
def classification_report_df(report):
    """Creates a data frame from sklearn.metrics.classification_report output.

    Parameters
    ------------
    report: sklearn classification report object
        Classification report object

    Returns
    --------
    Classification report DataFrame

    """
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = [splits for splits in line.split('  ') if splits is not ""]
        row_labels = ['class', 'precision','recall', 'f1_score', 'support']
        for idx, label in enumerate(row_labels):
            row[label] = row_data[idx]
        report_data.append(row)
    return pd.DataFrame(report_data)

def stability_stddev_of_scores(
        df, 
        cutoff_date, 
        SCORE_COLUMN, 
        ID_COLUMN = 'ACCOUNT_ID',
        SNAPSHOT_DT_COLUMN = 'SNAPSHOT_DT',
        compute_rolling_average = True,
        no_of_trailing_months = 12
    ):
    """Distribution plots of standard deviation of score_column

    Plot of std dev of::
        a) Difference in scores with previous month, per account
        b) Difference in moving average of the score with the previous month, per account 

    Parameters
    -----------
    df : Pandas dataframe
        DataFrame with atleast these three columns for example : 
        ``['ACCOUNT_ID', 'SNAPSHOT_DT', 'EWS_SCORE']``
    cutoff_date : date 
        Date from which plotting is done (>= cutoff_date)
    SCORE_COLUMN : string
        Column name of scores in the dataframe df
    ID_COLUMN: string, default: ``ACCOUNT_ID``
        The column in ``df`` that corresponds to an identifier (ex: ``ACCOUNT_ID`` or ``OPTY_ID`` etc.)
    SNAPSHOT_DT_COLUMN : string, default: ``SNAPSHOT_DT``
        The column in ``df`` that corresponds to a the snapshot date. 
    compute_rolling_average : boolean, default: ``True``
        When ``compute_rolling_average`` is ``True``, this function will compute a ``no_of_trailing_months``
        moving average of ``SCORE_COLUMN`` and recursively call this function with ``compute_rolling_average``
        set to ``False``.
    no_of_trailing_months : int, default: ``12``
        No of months to calculate rolling average. This argument is applicable only if ``compute_rolling_average``
        is ``True``, it will be ignored otherwise

    Returns
    --------
    Distribution plots, cumulative distribution plots and additionally the smoothed versions of the distribution, 
    cumulative distribution plots (rolling average over the last ``no_of_trailing_months``) 
    if ``compute_rolling_average`` is set to ``True``.

    """ 
    # Duplicate of the dataframe with just ACCOUNT_ID and SCORE columns
    # Use df.copy() otherwise (if you simply say df_duplicate = df, it will become a reference)
    df_prev = df[
                    [
                        ID_COLUMN, 
                        SNAPSHOT_DT_COLUMN, 
                        SCORE_COLUMN
                    ]
                ]
    # Creating offset of snapshot date by 1 month ahead to merge with original dataframe
    # so that we can obtain the score for the previous month for each month
    df_prev[SNAPSHOT_DT_COLUMN] = df_prev[SNAPSHOT_DT_COLUMN] + pd.offsets.DateOffset(months = 1)
    df_merged = df[
                    [
                        ID_COLUMN, 
                        SNAPSHOT_DT_COLUMN, 
                        SCORE_COLUMN
                    ]
                ].merge(
                        df_prev, 
                        on = [ID_COLUMN, SNAPSHOT_DT_COLUMN],
                        how = 'left',
                        suffixes = ('_CURR_MONTH', '_PREV_MONTH')
                    )
    # The suffix _CURR_MONTH denotes the SCORE_COLUMN from the dataframe on the left
    # and the suffix _PREV_MONTH denotes the SCORE_COLUMN from the dataframe on the right
    df_merged['SCORE_DIFF'] = np.abs(
                                    df_merged['{}_CURR_MONTH'.format(SCORE_COLUMN)] 
                                    -
                                    df_merged['{}_PREV_MONTH'.format(SCORE_COLUMN)]
                                )
    
    # Using cutoff_date to select instances >= cutoff_date 
    df_merged = df_merged[df_merged[SNAPSHOT_DT_COLUMN] >= pd.to_datetime(cutoff_date)]

    # Plotting std dev values
    stddev_df = df_merged.groupby(ID_COLUMN)[['SCORE_DIFF']].std().dropna()
    if(stddev_df.shape[0] > 2):
        stddev_df_values = stddev_df.values
        title = '\n'.join(
                wrap(
                    'Histogram of stddev of {} difference with previous months from {}'.format(
                        SCORE_COLUMN,
                        cutoff_date
                    ), 
                    50
                )
            )
        xlabel = 'stddev of {SCORE_COLUMN} diff with previous month per {ID_COLUMN}'.format(
                        SCORE_COLUMN = SCORE_COLUMN,
                        ID_COLUMN = ID_COLUMN
                    )

        distribution_plot(
                stddev_df_values, 
                title, 
                xlabel, 
                cumulative = True, 
                kde = True
            )
    else:
        logger.warning('Insufficient non-null samples for distribution plots')
    
    if(compute_rolling_average):
        # Using groupby in pandas on ID_COLUMN to calculate the rolling mean
        df_cpy = df[
                    [
                        ID_COLUMN,
                        SNAPSHOT_DT_COLUMN,
                        SCORE_COLUMN
                    ]
                ]
        df_rolling = pd.concat(
                            (
                                df_cpy[
                                    [
                                        ID_COLUMN, 
                                        SNAPSHOT_DT_COLUMN
                                    ]
                                ].sort_values(
                                    [
                                        ID_COLUMN, 
                                        SNAPSHOT_DT_COLUMN
                                    ]
                                ), 
                                df_cpy.sort_values(
                                        [
                                            ID_COLUMN, 
                                            SNAPSHOT_DT_COLUMN
                                        ]
                                    ).groupby(ID_COLUMN)[SCORE_COLUMN].apply(
                                            lambda x: x.rolling(
                                                    min_periods = 1, 
                                                    window = no_of_trailing_months
                                                ).mean()
                                        )
                            ), 
                            axis = 1
                         )
        SCORE_COLUMN_ROLLING_AVG = '{SCORE_COLUMN}_{no_of_trailing_months}_MONTH_ROLLING_AVG'.format(
                                            SCORE_COLUMN = SCORE_COLUMN,
                                            no_of_trailing_months = no_of_trailing_months
                                        )

        df_rolling = df_rolling.rename(columns = {SCORE_COLUMN : SCORE_COLUMN_ROLLING_AVG})
        # Recursive call
        stability_stddev_of_scores(
                df_rolling, 
                cutoff_date, 
                SCORE_COLUMN_ROLLING_AVG, 
                ID_COLUMN,
                SNAPSHOT_DT_COLUMN,
                # Do not compute rolling average
                False,
                no_of_trailing_months
            )
    return 


def plot_auc_cm_cr_for_multi_class(xgb_classifier, Xtest, ytest, ClfLabel):
    """Plots/prints AUC, Precision- recall curves, confusion matrix and classification report for multi class XGBoost classifier. 
    This works only for 4 classes.

    Parameters
    -----------    
    xgb_classifier: XGBoost multi class classifier
    	Any classifier with 4 outcome classes
    Xtest: Pandas dataframe
    	Test set feature vector
    ytest: Pandas dataframe
    	Test set actual outcomes
    ClfLabel: Class 
    	define all the actual outcomes using enum

    Returns
    --------
    xgb_classifier_y_prediction: Predicted probabilities for all instances
    cm: confusion matrix(for argmax(probability))
    classification_rep: classification report with all the four classes(for argmax(probability))
    AUC plot(4 classes plus micro AUC at all the thresholds)

    """
    xgb_classifier_y_prediction = xgb_classifier.predict_proba(
                                    Xtest, 
                                    ntree_limit = xgb_classifier.best_ntree_limit
                                )
    bin_ytest = preprocessing.label_binarize(ytest, classes = [e.value for e in ClfLabel])
    n_classes = bin_ytest.shape[1]
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(
                                            bin_ytest[:, i], 
                                            xgb_classifier_y_prediction[:, i], 
                                            pos_label = 1
                                        )
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], thresholds["micro"] = metrics.roc_curve(
                                                            bin_ytest.ravel(), 
                                                            xgb_classifier_y_prediction.ravel()
                                                        )
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    sns.set(style = 'darkgrid')
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label = 'micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
        color = 'deeppink', 
        linestyle = ':', 
        linewidth = 4
    )
    colors = sns.color_palette("Set1", n_classes)
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], 
            tpr[i], 
            color = color, 
            lw = 2,
            label = 'ROC curve of class {0} (area = {1:0.2f})'.format(ClfLabel(i).name, roc_auc[i])
        )
    plt.plot([0, 1], [0, 1], 'k--', lw = 2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: multi-class classification')
    plt.legend(loc = "lower right")
    plt.show()
    
    
    ### Precision - recall curve plotting 
    prec = dict()
    rec = dict()
    thresholds = dict()
    prec_recall = dict()
    for i in range(n_classes):
        prec[i], rec[i], thresholds[i] = metrics.precision_recall_curve(
                                            bin_ytest[:, i], 
                                            xgb_classifier_y_prediction[:, i], 
                                            pos_label = 1
                                        )
    prec["micro"], rec["micro"], thresholds["micro"] = metrics.precision_recall_curve(
                                                            bin_ytest.ravel(), 
                                                            xgb_classifier_y_prediction.ravel()
                                                        )
    plot_precision_recall_curve(prec["micro"], 
                                rec["micro"], 
                                thresholds["micro"], 
                                'micro-average Precision- Recall curve'
                                )
    for i in range(n_classes):
        plot_precision_recall_curve(prec[i], 
                                    rec[i], 
                                    thresholds[i], 
                                    'Precision- Recall curve of class {0}'.format(ClfLabel(i).name)
                                   )
    
    ### Confusion matrix caluclation
    ypred = np.argmax((xgb_classifier_y_prediction), axis = 1)
    cm = pd.DataFrame(
                 metrics.confusion_matrix(
                     ytest, 
                     ypred, 
                     labels = [e.value for e in ClfLabel]
                 ), 
                 [e.name for e in ClfLabel],
                 [e.name for e in ClfLabel]
            )
    cm1 = cm.copy()
    for m in range(len(ClfLabel)):
        for n in range(len(ClfLabel)):
            cm1.ix[m,n] = np.round((cm.ix[m,n]/sum(cm.ix[m,:])*100),1)

    #### ROWS ARE ACTUAL , COLUMNS ARE PREDICTED
    plt.figure(figsize = (10,5))
    ax = sns.heatmap(cm1, annot = True, annot_kws = {"size": 16}, fmt = '.1f')
    plt.title('Heatmap: Actual (rows) vs. Predicted (columns)')
    for t in ax.texts: 
        t.set_text(t.get_text() + " %")
    plt.show()

    classification_rep = metrics.classification_report(
                                ytest, 
                                ypred, 
                                target_names = [e.name for e in ClfLabel]
                            )
    classification_rep = classification_report_df(classification_rep)
    display(classification_rep)
    return(xgb_classifier_y_prediction, cm, classification_rep)

def show_prec_recall_f1score_stats(
        title, 
        df, 
        score_col,
        score_col_range,
        segment_col,
        segments,
        label_col,
        pos_label
    ):
    """
        Show precision-recall f1score stats
        
    Parameters
    ----------- 
    title: (string) title for current metric
    df: pandas dataframe containing the prediction dataset
    score_col: (string) name of the column in df that corresponds to the score/metric of interest
        The higher the score, the weaker the chance of pos_label 
    score_col_range: (float) max(df[score_col]) - min(df[score_col])
    segment_col: (string) name of the column in df that corresponds to the segment
    segments: (list) list of segments in the dataframe (sorted(df[segment_col].unique()))
    label_col: (string) name of the column in df that corresponds to the label
    pos_label: (string) the name of the positive class label in label_col
        
    Returns
    -----------
    Displays Precision-Recall,F1-score stats inline and returns a dataframe
    
    """
    display(Markdown('### {}'.format(title)))
    clf_metrics = []
    score_col_num_buckets = 20
    thresholds = [t*(score_col_range/score_col_num_buckets) for t in range(score_col_num_buckets + 1)]
    for t in thresholds:
        _df = df[[label_col, score_col]].dropna()
        y_true = (_df[label_col] == pos_label).astype(int)
        y_pred = (_df[score_col] <= t).astype(int)
        prec, rec, f1score, support = metrics.precision_recall_fscore_support(y_true, y_pred,  pos_label = 1)
        clf_metrics.append([t, prec[1], rec[1], f1score[1], support[1]])

    clf_metrics_df = pd.DataFrame(
                            clf_metrics, 
                            columns = [
                                        'threshold',
                                        'precision',
                                        'recall',
                                        'f1score',
                                        'support'
                                    ]
                        )
    display(clf_metrics_df)
    display(Markdown('#### Optimal threshold'))
    BEST_F1_SCORE = np.max(clf_metrics_df['f1score'])
    best_thresh_df = clf_metrics_df[clf_metrics_df['f1score'] == BEST_F1_SCORE]
    display(best_thresh_df)
    display(Markdown('#### Precision-Recall-F1 Scores by Segment at Optimal threshold'))
    clf_metrics_by_segment = []
    for segment in segments:
        if(segment == 'Overall'):
            _df = df[[label_col, score_col]].dropna()
        else:
            _df = df[df[segment_col] == segment][
                        [
                            label_col, 
                            score_col
                        ]
                    ].dropna()
        if(_df.shape[0] > 1):
            y_true = (_df[label_col] == pos_label).astype(int)
            y_pred = (_df[score_col] <= best_thresh_df['threshold'].values[0]).astype(int)
            prec, rec, f1score, support = metrics.precision_recall_fscore_support(y_true, y_pred,  pos_label = 1)
            clf_metrics_by_segment.append([segment, prec[1], rec[1], f1score[1], support[1]])

    clf_metrics_by_segment_df = pd.DataFrame(
                                    clf_metrics_by_segment, 
                                    columns = [
                                                'segment',
                                                'precision',
                                                'recall',
                                                'f1score',
                                                'support'
                                            ]
                                )
    display(clf_metrics_by_segment_df)
    return(clf_metrics_by_segment_df)
