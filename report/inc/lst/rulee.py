class Rule(object):
def __init__(self, antecedent=None, consequent=None, label=None,
             and_func=np.multiply, or_func=np.fmax):