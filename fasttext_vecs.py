""" Different functions regarding the fasttext vectors. The most relevant
function here is remove_upper(), which creates a new file that only contains
lower-cased words. """


import json
import io
import logging


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
  found, lower-case the word and add it to the new file. The first line of the
  file shows the number of entries and the number of embeddings. It has to 
  be updated after the procedure. """
  old = io.open(old_file, encoding='utf-8', newline='\n', errors='ignore')
  new = open(new_file, 'a', encoding='utf-8')
  low_words = get_lower(old_file)
  for line in old:
    if line[0].isupper():  # first char of line is upper-case
      tokens = line.rstrip().split(' ')
      low_token = tokens[0].lower()
      if low_token not in low_words:
        logging.info(f'{tokens[0]} has no lower-cased equivalent')
        low_line = ' '.join([low_token] + tokens[1:])
        new.write(low_line)
    else:
      new.write(line)
  new.close()
  lines = open(new_file, encoding='utf-8').readlines()
  logging.info(f'New file has {len(lines)-1} words')
  lines[0] = f'{len(lines)-1} {lines[0].rstrip().split(" ")[1]}'
  with open(new_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)


def get_lower(fname):
  """Return all the lower words in the given IO object as a list. """
  vecs = io.open(fname, encoding='utf-8', newline='\n', errors='ignore')
  words = []
  for line in vecs:
    if not line[0].isupper():
      words.append(line.rstrip().split(' ')[0])
  logging.info(f'{len(words)} non-capitalized words found in the file')
  return words


def vocab_vectors(vecs_file, vocab_file, dump_file):
  """ Dump all vectors whose words are in the vocab. The first line of the
  file shows the number of entries and the number of embeddings. It has to 
  be updated after the procedure."""
  vecs = io.open(vecs_file, encoding='utf-8', newline='\n', errors='ignore')
  vocab = list(json.load(open(vocab_file)).keys())
  logging.info(f'Vocab has {len(vocab)} words')
  dump = open(dump_file, 'a', encoding='utf-8')
  dump.write(vecs.readline())  # dump first line with file info.
  for line in vecs:
    word = line.split(' ')[0]
    if word in vocab:
      vocab.remove(word)
      dump.write(line)
  dump.close()
  logging.info(vocab)
  logging.info(f'{len(vocab)} vocab words are not in the vector file')
  lines = open(dump_file, encoding='utf-8').readlines()
  logging.info(f'{len(lines)-1} lines in the new file')
  lines[0] = f'{len(lines)-1} {lines[0].rstrip().split(" ")[1]}'
  with open(dump_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)
  

if __name__ == '__main__':
  vecs_file = 'data/pretrained_vecs/wiki-news-300d-1M-subword.vec'
  lower_file = 'data/pretrained_vecs/lower_vecs.vec'
  vocab_file = 'data/pretrained_vecs/vocab_vecs.vec'
  vocab_json = 'data/vocab/vocab.json'
  logging.basicConfig(
    level=logging.INFO, 
    handlers=[logging.FileHandler('logs/vocab_vectors.log', 'w', 'utf-8')],
    format='%(message)s'
  )
  vocab_vectors(vecs_file, vocab_json, vocab_file)
  # vocab_vectors(lower_file, vocab_file, dump_file)
