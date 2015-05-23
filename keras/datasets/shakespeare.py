"""Download Project Gutenburg Shakespeare's Full-text"""
from . import data_utils
BASE_URL = 'http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt'
TARGET_FILENAME = 'shakespeare.txt'

def load_data(path='shakespeare.txt', url=BASE_URL):
    """Load the shakespeare input dataset for rnn-effectiveness
    
    http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    
    returns downloaded multi-megabyte file...
    """
    filename = data_utils.get_file( path, BASE_URL )
    content = open(filename).read()
    # yes, there are faster ways, it's a small dataset...
    for junk in [
        '\xbb',
         '\xbf',
         '\xef'
    ]:
        content = content.replace(junk, '')
    chars = set(content)
    return content, chars
