from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def generate_tokenizer(equations, output, vocab_size):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(special_tokens=["[PAD]", "[BOS]", "[EOS]"], vocab_size=vocab_size, show_progress=True)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(trainer, [equations])
    tokenizer.save(path=output, pretty=False)

if __name__ == '__main__':

    equations = './dataset/math.txt'
    output = './dataset/tokenizer.json'
    vocab_size = 1000

    generate_tokenizer(equations, output, vocab_size)
