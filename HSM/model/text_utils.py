import spacy
from bs4 import BeautifulSoup
import unicodedata
from string import punctuation
import re
import contractions
import os
import pandas as pd
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class NormalizeText():
    """
    Description:
        The class's transform method will normalize text by stripping html
        tags; replacing accented characters with their utf-8 counterparts;
        expanding contractions; removing stopwords; nixing words that aren't at
        least 3 characters long or less than 17 chars long and only alphabetic;
        lemmatizing words; replacing profanity with the string 'criticaster';
        replacing urls, ssns, email addresses and phone numbers with the word
        'blatherskite'; and finally converting any empty normalized strings to
        the string 'spam'.
    """

    def __init__(self):
        self.nlp = spacy.load('en', parse = False, tag=False, entity=False)


    def fit(self, X_train, y=None):
        return self

    @staticmethod
    def strip_html_tags(text):
        """
        Given a string, use bs4 to strip html tags.
        """
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    @staticmethod
    def replace_accented_chars(text):
        """
        Given a string, replace accented characters with their non-accented
        counterparts using the normal form KD (NFKD).
        """
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    @staticmethod
    def expand_contractions(text):
        """
        Given a string, replace contrctions with their expanded counterparts.
        """
        contraction_mapping = contractions.contractions_dict
        contractions_list = contraction_mapping.keys()
        contractions_pattern = re.compile('({})'.format('|'.join(contractions_list)),
                                          flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            if contraction_mapping.get(match):
                expanded_contraction = contraction_mapping.get(match)
            else:
                expanded_contraction = contraction_mapping.get(match.lower())
            if expanded_contraction:
                expanded_contraction = first_char+expanded_contraction[1:]
                return expanded_contraction
            else:
                pass

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text


    def lemmatize_text(self,text):
        """
        Given a list of words, lemmatize them.
        """
        text = self.nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text

    @staticmethod
    def remove_stopwords(text, is_lower_case=True):
        """
        Given a string, tokenize and remove stopwords.
        """
        stopword_list = stopwords.words('english')
        #we want the negatives
        stopword_list.remove('no')
        stopword_list.remove('not')
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text


    def normalize_text(self, doc, html_stripping=True, contraction_expansion=True,
                         text_lemmatization=True, stopword_removal=True):
        """
        Apply all of the text normalization methods to a collection of docs
        within a corpus.
        """

        def get_profanity():
            """
            Read in a list of profanity from profanity.csv and return it as a
            set.
            """
            file_path = os.path.join(os.getcwd(),"corpora","profanity.csv")
            profanity = set(pd.read_csv(file_path).values.ravel().tolist())
            return profanity


        url_re = re.compile(r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""")
        email_re = re.compile(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)')
        phone_re = re.compile(r'(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?')
        ssn_re = re.compile(r'^(?!219-09-9999|078-05-1120)(?!666|000|9\d{2})\d{3}[-]?(?!00)\d{2}[-]?(?!0{4})\d{4}$')
        profanity_regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, get_profanity())))

        doc = doc.lower()
        doc = profanity_regex.sub("criticaster", doc)
        doc = email_re.sub('blatherskite',doc)
        doc = phone_re.sub('blatherskite',doc)
        doc = ssn_re.sub('blatherskite',doc)
        doc = url_re.sub('blatherskite',doc)

        # strip HTML
        if html_stripping:
            doc = self.strip_html_tags(doc)
        # expand contractions
        if contraction_expansion:
            doc = self.expand_contractions(doc)
        # at least 3 chars long, no numbers, and no more than 17 chars long
        doc = re.findall(r'\b[a-z][a-z][a-z]+\b',doc)
        doc = ' '.join(w for w in doc if w != 'nan' and len(w) <= 17)
        # lemmatize text
        if text_lemmatization:
            doc = self.lemmatize_text(doc)
        # remove stopwords
        if stopword_removal:
            doc = self.remove_stopwords(doc)
        if len(doc) == 0:
            doc = "spam"
        normalized_doc = doc
        return normalized_doc


    def transform(self,Xtrain,y=None):
        normalized_text = Xtrain.apply(self.normalize_text)
        return normalized_text
