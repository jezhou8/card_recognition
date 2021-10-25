# a function that calls https://api.dictionaryapi.dev/api/v2/entries/en/<word> and returns the response as json
# <word> is the word to be searched
# return score divided by length of word_list
def similarity(word, word_list):
    import requests
    url = 'https://api.dictionaryapi.dev/api/v2/entries/en/' + word
    response = requests.get(url)

    score = 0
    # iterate through meanings array in the response
    for meaning in response.json()[0]['meanings']:
        # iterate through definitions array in the meaning
        for definition in meaning['definitions']:
            # if synonyms exist, iterate through them
            if 'synonyms' in definition:
                for synonym in definition['synonyms']:
                    # if the synonym in the word list, add 1 to score
                    if synonym in word_list:
                        score += 1
            # if antonyms exist, iterate through them
            if 'antonyms' in definition:
                for antonym in definition['antonyms']:
                    # if the antonym in the word list, subtract 1 from score
                    if antonym in word_list:
                        score -= 1
            
    return score / len(word_list)


word = "smart"

word_list = ["clever", "bright", "stupid", "silly"]

print(similarity(word, word_list))