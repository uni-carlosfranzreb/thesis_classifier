""" Load fasttext vectors from the text file. It is over 2GB big: it should
be fine for the server. """


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


if __name__ == '__main__':
  vecs_file = 'data/pretrained_vecs/wiki-news-300d-1M-subword.vec'
  # vecs = sample_vectors(vecs_file)
  # json.dump(vecs, open('sample_vecs.json', 'w'))
  for word in ['machine', 'machines', 'learn', 'learning', 'learns', 'learned']:
    print(word, find_vec(vecs_file, word) is not None)
  for word in ['Machine', 'Machines', 'Learn', 'Learning', 'Learns', 'Learned']:
    print(word, find_vec(vecs_file, word) is not None)