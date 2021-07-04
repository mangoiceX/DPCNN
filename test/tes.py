# from gensim.models import FastText
# from gensim.test.utils import common_texts  # some example sentences
# print(common_texts)
# model = FastText(vector_size=4, window=3, min_count=1)  # instantiate
# model.build_vocab(corpus_iterable=common_texts)
# model.train(corpus_iterable=common_texts, total_examples=len(common_texts), epochs=10)  # train
# # sentences_1 = [["cat", "say", "meow"], ["dog", "say", "woof"]]
# # sentences_2 = [["dude", "say", "wazzup!"]]

# # model = FastText(min_count=1)
# # model.build_vocab(sentences_1)
# # model.train(sentences_1, total_examples=model.corpus_count, epochs=model.iter)

# # model.build_vocab(sentences_2, update=True)
# # model.train(sentences_2, total_examples=model.corpus_count, epochs=model.iter)

from gensim.test.utils import get_tmpfile
fname = get_tmpfile("fasttext.model")
print(fname)
