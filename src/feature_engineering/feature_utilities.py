import pandas as pd
import numpy as np
import datetime


# Returns a tuple:
# - type
# - default values
# - replace default values with
# - special encoder to use, if any
def identify_feature(feature_data):
    feature_type = feature_data.dtype.name

    if feature_type == 'object':
        return identify_feature_object(feature_data)
    elif feature_type == 'float64':
        return identify_feature_float(feature_data)
    elif feature_type == 'int64':
        return identify_feature_int(feature_data)
    else:
    #     #raise ValueError('Unknown feature type')
        return 'Unknown', 'Unknown', [], 0, 'Standard'

MAX_VALUES_STANDARD_FEATURE = 15


def transform_date(value):
    date = parse_date(value)

    if date is None:
        return np.nan
    else:
        return (date - datetime.datetime(2000, 1, 1)).days


def parse_date(value):
    if type(value) != str or len(value) != 16:
        return None

    try:
        date = pd.datetime.strptime(value, '%d%b%y:%H:%M:%S')
        return date
    except:
        return None


# TODO: Analyze for 0 ?
def is_int_score(feature_data_values):
    if not 1 < feature_data_values.shape[0] < 110:
        return None

    score_threshold = 0.5
    if feature_data_values.shape[0] > 5:
        score_threshold = 0.75
    # Check that the values mostly fit
    if not feature_data_values.index[feature_data_values.index < feature_data_values.shape[0]].shape[0] / float(feature_data_values.shape[0]) > score_threshold:
        return None

    # Find the default values
    default_values = []
    default_values.extend(feature_data_values.index[feature_data_values.index < 0])

    if feature_data_values.shape[0] < 70:
        default_values.extend(feature_data_values.index[feature_data_values.index > 90])
    else:
        default_values.extend(feature_data_values.index[feature_data_values.index > 900])

    return 'Int', 'Score', default_values, -1, None


def is_float_score(feature_data_values):
    if not 1 < feature_data_values.shape[0] < 110:
        return None

    score_threshold = 0.5
    if feature_data_values.shape[0] > 5:
        score_threshold = 0.75
    # Check that the values mostly fit
    if not feature_data_values.index[feature_data_values.index < feature_data_values.shape[0]].shape[0] / float(feature_data_values.shape[0]) > score_threshold:
        return None

    # Find the default values
    default_values = []
    default_values.extend(feature_data_values.index[feature_data_values.index < 0])

    if feature_data_values.shape[0] < 70:
        default_values.extend(feature_data_values.index[feature_data_values.index > 90])
    else:
        default_values.extend(feature_data_values.index[feature_data_values.index > 900])

    return 'Float', 'Score', default_values, -1, None




# ---------------------------------- 'Generic' # ----------------------------------

def identify_feature_object(feature_data):
    values = feature_data.value_counts()

    if len(values) < 2:
        return 'Object', 'Constant', [], -1, 'Standard'

    # Date - simple - take the top 5...
    is_date = True
    for i in range(0, min(len(values), 5)):
        if values.index[i] == 'nan':
            continue
        if parse_date(values.index[i]) is None:
            is_date = False
            break

    if is_date:
        return 'Object', 'Date', [], -1, None

    # Standardized (U, H, R, Q)
    if 1 < values.shape[0] <= MAX_VALUES_STANDARD_FEATURE and values.index[0] in ['U', 'O', 'Q', 'R', 'H', '-1', 'B', 'C', 'N', 'S', 'I'] and values.index[1] in ['U', 'O', 'Q', 'R', 'H', '-1', 'B', 'C', 'N', 'S', 'I']:
        return 'Object', 'Standard', ['-1'], -1, None

    if 0 < values.shape[0] <= 2 and values.index[0] in [False, True]:
        return 'Object', 'Boolean', [], -1, None

    # State - top 5 are either -1 or 2 letters
    is_state = True
    for i in range(0, min(len(values), 5)):
        if len(values.index[i]) != 2:
            is_state = False

    if is_state:
        return 'Object', 'State', ['-1'], -1, None

    # Occupation etc (maybe a simple -> empty vs not empty ?)
    return 'Object', 'Unknown', [], -1, None


def identify_feature_float(feature_data):
    values = feature_data.value_counts()

    ret = is_float_score(values)
    if ret is not None:
        return ret

    return 'Float', 'Unknown', [], -1, 'Standard'


def identify_feature_int(feature_data):
    values = feature_data.value_counts()

    ret = is_int_score(values)
    if ret is not None:
        return ret

    return 'Int', 'Unknown', [], -1, 'Standard'
