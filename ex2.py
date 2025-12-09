import sys
import math
from collections import Counter
from file_utils import read_file_lines, write_outputs_to_file, Output

VOCAB_SIZE = 300000


def make_dict(words: list[str]) -> dict[str, int]:
    return dict(Counter(words))


def split_to_words(lines: list[str]) -> list[str]:
    words = []
    for line in lines:
        line_words = line.split()
        for word in line_words:
            words.append(word)    
    return words

class Sample:
    def __init__(self, events: list[str]):
        self.events = events
        self.size = len(events)
        self.dict: dict[str, int] = make_dict(events)
        self.vocab_size = VOCAB_SIZE

    def unique_events_count(self):
        return len(self.dict.keys())
    
    def count_word(self, word: str) -> int:
        return self.dict.get(word, 0)
    
    def mle(self, word: str) -> float:
        return self.dict.get(word, 0) / self.size
    
    def lidstone_mle(self, word: str, _lambda: float) -> float:
        return (self.dict.get(word, 0) + _lambda) / (self.size + _lambda * self.vocab_size)


def perplexity(sample: Sample, prob_func) -> float:
    log_sum = 0
    for word in sample.events:
        prob = prob_func(word)
        if prob > 0:
            log_sum += -math.log2(prob)    
    return 2 ** (log_sum / sample.size)    

# Parse command line arguments
if len(sys.argv) != 5:
    print("Error: Expected 4 arguments")
    print("Usage: python args_parser.py <dev_file_name> <test_file_name> <input_word> <output_file_name>")
    sys.exit(1)

dev_file_name = sys.argv[1]
test_file_name = sys.argv[2]
input_word = sys.argv[3]
output_file_name = sys.argv[4]


outputs: list[Output] = []

outputs.append(Output(1, [dev_file_name]))
outputs.append(Output(2, [test_file_name]))
outputs.append(Output(3, [input_word]))
outputs.append(Output(4, [output_file_name]))
outputs.append(Output(5, [VOCAB_SIZE]))
outputs.append(Output(6, [1/VOCAB_SIZE]))


dev_lines = read_file_lines(dev_file_name)
dev_words = split_to_words(dev_lines)

num_of_90_percent = round(len(dev_words) * 0.9)

training_set = Sample(dev_words[:num_of_90_percent])
validation_set = Sample(dev_words[num_of_90_percent:])

outputs.append(Output(7, [str(len(dev_words))]))
outputs.append(Output(8, [str(validation_set.size)]))
outputs.append(Output(9, [str(training_set.size)]))


outputs.append(Output(10, [str(training_set.unique_events_count())]))
outputs.append(Output(11, [str(training_set.count_word(input_word))]))
outputs.append(Output(12, [str(training_set.mle(input_word))]))
outputs.append(Output(13, [str(training_set.mle('unseen-word'))]))
outputs.append(Output(14, [str(training_set.lidstone_mle(input_word, 0.1))]))
outputs.append(Output(15, [str(training_set.lidstone_mle('unseen-word', 0.1))]))
outputs.append(Output(16, [str(perplexity(validation_set, lambda word: training_set.lidstone_mle(word, 0.01)))]))
outputs.append(Output(17, [str(perplexity(validation_set, lambda word: training_set.lidstone_mle(word, 0.1)))]))
outputs.append(Output(18, [str(perplexity(validation_set, lambda word: training_set.lidstone_mle(word, 1)))]))

write_outputs_to_file(output_file_name, outputs)
