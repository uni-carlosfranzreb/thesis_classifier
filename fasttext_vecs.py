""" Different functions regarding the fasttext vectors. The most relevant
function here is remove_upper(), which creates a new file that only contains
lower-cased words. """


import json
import io


def load_vectors(fname):
  """ Function from the fasttext website. """
  fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
  data = {}
  for line in fin:
    tokens = line.rstrip().split(' ')
    data[tokens[0]] = map(float, tokens[1:])
  return data


def sample_vectors(fname, n=50):
  """ Extract the first n vecs of the file for inspection and testing. """
  fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
  data = {}
  cnt = 0
  for line in fin:
    tokens = line.rstrip().split(' ')
    data[tokens[0]] = list(map(float, tokens[1:]))
    cnt += 1
    if cnt == n:
      return data


def find_vec(fname, token):
  """ Return the vector for the given token or None if not found. """
  fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
  for line in fin:
    if line[:len(token)+1] == token + ' ':
      tokens = line.rstrip().split(' ')
      return list(map(float, tokens[1:]))


def test_embeddings(fname):
  """ See what kind of words are in the file, i.e. if words are lemmatized or 
  lower-cased. """
  for word in ['machine', 'machines', 'learn', 'learning', 'learns', 'learned']:
    print(word, find_vec(fname, word) is not None)
  for word in ['Machine', 'Machines', 'Learn', 'Learning', 'Learns', 'Learned']:
    print(word, find_vec(fname, word) is not None)


def find_ngrams(fname, n=20):
  """ Return the first n n-grams found in the file. Words of the n-grams are
  joined with underscores. """
  fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
  cnt = 0
  for line in fin:
    word = line.rstrip().split(' ')[0]
    if '_' in word:
      print(word)
      cnt += 1
      if cnt == n:
        return


def remove_upper(old_file, new_file):
  """ Remove all upper-cased words from the vec file. To ensure that no words
  lost, look for the lower-cased version of the word in the file. If it is not
  found, lower-case the word and add it to the new file. """
  old = io.open(old_file, encoding='utf-8', newline='\n', errors='ignore')
  new = open(new_file, 'a', encoding='utf-8')
  for line in old:
    if line[0].isupper():
      tokens = line.rstrip().split(' ')
      low_token = tokens[0].lower()
      if find_vec(old_file, low_token) is None:
        low_line = ' '.join([low_token] + tokens[1:])
        new.write(low_line)
    else:
      new.write(line)


if __name__ == '__main__':
  vecs_file = 'data/pretrained_vecs/wiki-news-300d-1M-subword.vec'
  new_file = 'data/pretrained_vecs/lower.vec'
  remove_upper(vecs_file, new_file)
