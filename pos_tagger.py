"""POS_taggers
"""
import nltk
from pattern.en import tag

class POStagger(object):
    """POS tagger class

    ## start with initial tagger like
    tagger = POStagger('nltk')

    ## switch tagger
    tagger.switch_tagger('hermes')

    ## tagging
    tagger.tagging('I want to eat breakfast')

    """
    def __init__(self, initial_tagger):
        self.selected_tagger = initial_tagger
        self.tagging = self.import_tagger()

    def import_tagger(self):
        """ import tagger
        """
        if self.selected_tagger == 'nltk':
            return nltk_tagging
        elif self.selected_tagger == 'pattern':
            return pattern_tagging
        # TODO(geonho): add other POS taggers
        else:
            print "ERROR: Wrong tagger has been imported"

    def switch_tagger(self, new_tagger):
        """Change POS tagger
        """
        self.selected_tagger = new_tagger
        self.tagging = self.import_tagger()

    def tagging(self, sentence_to_tagging):
        """Call tagging of selected POS tagger
        """
        return self.tagger(sentence_to_tagging)


def nltk_tagging(sentence):
    """NLTK pos tagging function
    """
    word_tokens = nltk.word_tokenize(sentence)
    # TODO(geonho): should define regular form of POS tagged output
    return nltk.pos_tag(word_tokens)

def pattern_tagging(sentence):
    """Pattern pos tagging function
    """
    return tag(sentence)

