"""Download Project Gutenburg Shakespeare's Full-text"""
import re
from . import data_utils
BASE_URL = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
TARGET_FILENAME = 'shakespeare.txt'

KILL_LICENSE_NOTICES = re.compile(r"^\<\<.*?\>\>",re.M|re.I|re.U|re.DOTALL)
SPLIT_TEXTS = re.compile( "^THE END", re.M|re.U )

def cleanup( content ):
    """Strip out the parts of the document that aren't Shakespeare's writings"""
    content = content[content.index('1609'):]
    content = KILL_LICENSE_NOTICES.sub("", content)
    content = content[:content.index('End of the Project Gutenberg EBook of The Complete Works of William')]
    texts = [
        text.strip()
        for text in SPLIT_TEXTS.split(content)[:-1]
    ]
    return texts

def load_data(path='shakespeare.txt', url=BASE_URL):
    """Load the text (from the internet the first time) and split into the texts
    
    There are 37 texts in the file, starting with the Sonnets
    
    returns multi-line unicode texts
    """
    filename = data_utils.get_file( path, BASE_URL )
    content = open(filename,'rb').read().decode('utf-8')
    texts = cleanup(content)
    return texts
