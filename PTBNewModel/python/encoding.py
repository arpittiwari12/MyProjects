#Methods to process aov band strings 

def to_decimal(x, col='Marketing Cloud Social AOV'):
    from decimal import Decimal
    res = re.sub(r'[\D.]','', x[col])
    # Searching for the same pattern (should match only once)
    res = re.findall(r'[\d+.,]+', res)
    if res is None or len(res) > 1:
        raise Exception('More than one number was found: %s' %res)
    else:
        return Decimal(res[0])
    
def band_to_number (x, col, method='mean'):
    import re
    """Converts an AOV_band to a number based on the chosen method. Defualt is median of the band"""
    try:
        nums = re.split('-',re.sub(r'[\$+]','', re.sub(r'[Kk]','000', re.sub(r'[Mm]','000000', x[col]))))
        low = float(nums[0])
    except Exception:
        return None
    try:
        high = float(nums[1])
    except Exception:
        high = low
    mean = (low + high)/2
    return mean
    
    
def to_float(x, col):
    return float(re.sub(r'[^\d.]', '', x[col]))


def encode_clouds(input):
    import pandas as pd
    vector = {
        'sales': 0,
        'custom': 0,
        'communities':0,
        'data':0,
        'other':0,
        'service':0,
        'collaboration':0,
        'else': 0
    }
    for token in input.strip().lower().split():
        token = token.strip()
        if token == 'and' : continue
        elif token == 'sales': vector['sales'] = 1
        elif token == 'custom': vector['custom'] = 1
        elif token == 'communities': vector['communities'] = 1
        elif token == 'data': vector['data'] = 1
        elif token == 'service': vector['service'] = 1
        elif token == 'collaboration': vector['collaboration'] = 1
        elif token == 'other': vector['data'] = 1
        else: vector['else'] = 1
    return pd.Series(vector)


def encode_editions(text):
    
    text = text.lower().strip()
    if text == "px" or text == "ue": return 5
    elif text == "ee": return 4
    elif text == "pe": return 3
    elif text == "ge" or text == "ce": return 2
    elif text is None: return 1
    else: return 0


def encode_premier(text):
    
    if text is None: return None
    else:
        text = text.strip()
    if text == 'Premier + Admin Top Customer': return 6
    elif text == 'Premier + Admin': return 5
    elif text == 'Premier + Admin < 50': return 4
    elif text == 'Premier': return 3
    elif text == 'Premier < 50': return 2
    elif text == 'Basic': return 1
    elif text == 'None': return 1 #seems like just a basic but missing the value
    else: return None
 
                                                   
                  
def encode_boolean(text, default=0):
    if text is None: return None
    else:
        text = text.lower().strip()
    if text == 'false': return 0
    elif text == 'true': return 1
    else: return default
