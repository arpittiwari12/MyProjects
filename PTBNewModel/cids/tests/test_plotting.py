import uuid
import nose.tools
from nose.tools import assert_equals, raises, nottest
import matplotlib
matplotlib.use('agg')
from ..plotting import *
plot_train_test_curve = nottest(plot_train_test_curve)



def _create_plot_auc_cm_cr_for_multi_class():
    """Helper function to create mock datasets"""
    TRAIN_SET_SIZE = 500
    HOLD_OUT_SET_SIZE = 1000
    NUM_OF_METRICS = 100
    FEATURE_COLUMNS = ['METRIC_{}'.format(i) for i in range(NUM_OF_METRICS)]
    Xtrain = pd.DataFrame(np.random.random((TRAIN_SET_SIZE, NUM_OF_METRICS)), columns = FEATURE_COLUMNS)
    Xtest = pd.DataFrame(np.random.random((HOLD_OUT_SET_SIZE, NUM_OF_METRICS)), columns = FEATURE_COLUMNS)
    ytrain = [np.random.randint(0, len(EWS360ClassLabels)) for num in range(TRAIN_SET_SIZE)]
    ytest = [np.random.randint(0, len(EWS360ClassLabels)) for num in range(HOLD_OUT_SET_SIZE)]
    eval_set = [(Xtrain, ytrain), (Xtest, ytest)]
    xgb_classifier_mdl = xgboost.XGBClassifier(n_estimators = 50)
    xgb_classifier_mdl.fit(Xtrain, 
                           ytrain, 
                           eval_set = eval_set, 
                           early_stopping_rounds = 0.10*50
                           )
    predictions, \
    cm, \
    report = plot_auc_cm_cr_for_multi_class(xgb_classifier_mdl, Xtest, ytest, EWS360ClassLabels)
    return(ytest, predictions, cm, report)

def test_plot_auc_cm_cr_for_multi_class():
    """Test plotting auc and generating confusion matrix function in plotting.py"""
    ytest, \
    predictions, \
    cm, \
    report = _create_plot_auc_cm_cr_for_multi_class()

    assert_equals(
        np.all(predictions.ravel()) >= 0, 
        True, 
        'Some of the probabilities are less than 0'
        )

    assert_equals(
        np.all(predictions.ravel()) <= 1, 
        True, 
        'Some of the probabilities are greater than 1'
        )

    assert_equals(
        len(predictions) > 0, 
        True, 
        'The classifier did not make any predictions'
        )

    assert_equals(
        len(predictions), 
        len(ytest), 
        'Not all instances in the test set are used for predictions'
        )

    assert_equals(
        len(predictions), 
        report['support'].astype(int).sum(), 
        'There are some missing classes that are not listed in the classification report'
        )

    assert_equals(
        np.all([cm.ix[i,:].sum() == len([x for x in ytest if x == i]) for i in range(len(predictions[0]))]), 
        True, 
        'Confusion matrix is not properly created'
        )
