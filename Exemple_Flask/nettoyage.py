from unidecode import unidecode
import re
from nltk.stem import SnowballStemmer

def nettoyage(string):
    l=[]
    string=unidecode(string.lower())
    string=" ".join(re.findall("[a-zA-Z]+", string))
    fr = SnowballStemmer('french')
    for word in string.split():
            l.append(fr.stem(word))
    return ' '.join(l)