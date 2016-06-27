class Vocabulary:
	def __init__(self):
		self.vocabulary = {"<UNK>": 0, "<EOS>": 1}
		self.id_list = {0: "<UNK>", 1: "<EOS>"}
		self.size = 2

	def load(self, file):
		with open(file, "r") as f:
			for word in f:
				word = word.strip()
				if word not in self.vocabulary:
					self.vocabulary[word] = len(self.vocabulary)
					self.id_list[len(self.vocabulary) - 1] = word
		self.size = len(self.vocabulary)

	def w2id(self, word):
		if word in self.vocabulary.keys():
			return self.vocabulary[word]
		else:
			return self.vocabulary["<UNK>"]

	def id2w(self, id):
		return self.id_list[id]

	def size(self):
		return self.size





	def test(self):
		for i in range(10):
			print(self.id_list[i], self.vocabulary[self.id_list[i]])
		return