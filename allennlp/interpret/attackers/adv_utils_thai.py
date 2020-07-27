from copy import copy, deepcopy

consonant="กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮ"
front_vowel="เแโใไ"
lower_vowel="อุอู".replace('อ','') #ทำแบบนี้จะได้อ่านออก
rear_vowel = "าําๅๆะฯๅๆ"
upper_vowel = "อ็อ้อ์อิอีอือึอํอัอ่อ๋อ๊".replace('อ','')
tone = "อ้อ่อ๋อ๊".replace('อ','')

CHARACTERS = consonant+front_vowel+lower_vowel+rear_vowel+upper_vowel+tone

def add_char(w_original):
    """
    add letter in the middle of a word
    """
    word_list = []
    if len(w_original) > 3:
        for i in range(1, len(w_original)-1):
            for char in CHARACTERS:
                w = copy(w_original)
                w = list(w)
                w[i] = char + w[i]
                word_list.append("".join(w))
        return word_list
    else:
        return [w_original]
    
def del_char(w_original):
    """
    del letter except for the first letter and the last letter
    """
    word_list = []
    if len(w_original) > 3:
        for i in range(1, len(w_original)-1):
            w = copy(w_original)
            w = list(w)
            w[i] = ''
            word_list.append("".join(w))
        return list(set(word_list))
    else:
        return [w_original]
    
def swap_char(w_original):
    """
    del letter except for the first letter and the last letter
    """
    word_list = []
    if len(w_original) > 3:
        for i in range(1, len(w_original)-1):
            w = copy(w_original)
            w = list(w)
            w[i],w[i+1] = w[i+1],w[i]
            word_list.append("".join(w))
        return list(set(word_list))
    else:
        return [w_original]
    
def replace_char(w_original):
    """
    add letter in the middle of a word
    """
    word_list = []
    if len(w_original) > 3:
        for i in range(1, len(w_original)-1):
            for char in CHARACTERS:
                w = copy(w_original)
                w = list(w)
                w[i] = char 
                word_list.append("".join(w))
        return list(set(word_list))
    else:
        return [w_original]



def get_adversarial_candidates(w_original):
    out=add_char(w_original)+del_char(w_original)+swap_char(w_original)+replace_char(w_original)
    return list(set(out))

