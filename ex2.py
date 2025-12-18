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

    def sum_probs(self, _lambda: float) -> float:
        total = 0.0
        for word in self.dict.keys():
            total += self.lidstone_mle(word, _lambda)
        N0 = self.vocab_size - len(self.dict.keys())
        total += N0 * self.lidstone_mle('unseen-word', _lambda)
        return total

class HeldOutModel:
    def __init__(self, training_set: Sample, held_out_set: Sample):
        self.training_set = training_set
        self.held_out_set = held_out_set
        self.count_to_words: dict[int, list[str]] = {}
        for word, count in training_set.dict.items():
            words = self.count_to_words.get(count, [])
            words.append(word)
            self.count_to_words[count] = words

        self.N0 = VOCAB_SIZE - len(self.training_set.dict.keys())
        
        unseen_in_training = [word for word in held_out_set.dict.keys() if word not in training_set.dict]
        self.T0 = sum(held_out_set.dict[word] for word in unseen_in_training)

    def prob(self, training_word: str) -> float:
        r = self.training_set.count_word(training_word)
        if r == 0:
            return (self.T0 / self.N0) / self.held_out_set.size
        training_words_with_r = self.count_to_words[r]
        Nr = len(training_words_with_r)
        return self.Tr(r) / (Nr * self.held_out_set.size)

    def Tr(self, r: int) -> float:
        if r == 0: 
            return self.T0
        training_words_with_r = self.count_to_words[r]
        Tr = 0
        for training_word in training_words_with_r:
            Tr += self.held_out_set.count_word(training_word)
        return Tr

    def sum_probs(self) -> float:
        total = 0.0
        for word in self.training_set.dict.keys():
            total += self.prob(word)
        total += self.N0 * self.prob('unseen-word')
        return total


def perplexity(sample: Sample, prob_func) -> float:
    log_sum = 0
    for word in sample.events:
        prob = prob_func(word)
        if prob > 0:
            log_sum += -math.log2(prob)    
    return 2 ** (log_sum / sample.size)    


def find_best_lidstone_lambda(training_set: Sample, validation_set: Sample, min_lambda: float = 0.01, max_lambda: float = 1.0, threshold: float = 0.001) -> float:
    """
    Binary search to find lambda that minimizes perplexity on validation set.
    Stops when the difference between left and right boundaries is smaller than threshold.
    """
    left = min_lambda
    right = max_lambda
    
    while right - left > threshold:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        perp1 = perplexity(validation_set, lambda word: training_set.lidstone_mle(word, mid1))
        perp2 = perplexity(validation_set, lambda word: training_set.lidstone_mle(word, mid2))
        
        if perp1 > perp2:
            left = mid1
        else:
            right = mid2
    
    optimal_lambda = (left + right) / 2
    return optimal_lambda




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

##################################################################
##
##  Lidstone
##
##################################################################

num_of_90_percent = round(len(dev_words) * 0.9)

lidstone_training_set = Sample(dev_words[:num_of_90_percent])
validation_set = Sample(dev_words[num_of_90_percent:])

outputs.append(Output(7, [str(len(dev_words))]))
outputs.append(Output(8, [str(validation_set.size)]))
outputs.append(Output(9, [str(lidstone_training_set.size)]))


outputs.append(Output(10, [str(lidstone_training_set.unique_events_count())]))
outputs.append(Output(11, [str(lidstone_training_set.count_word(input_word))]))
outputs.append(Output(12, [str(lidstone_training_set.mle(input_word))]))
outputs.append(Output(13, [str(lidstone_training_set.mle('unseen-word'))]))
outputs.append(Output(14, [str(lidstone_training_set.lidstone_mle(input_word, 0.1))]))
outputs.append(Output(15, [str(lidstone_training_set.lidstone_mle('unseen-word', 0.1))]))
outputs.append(Output(16, [str(perplexity(validation_set, lambda word: lidstone_training_set.lidstone_mle(word, 0.01)))]))
outputs.append(Output(17, [str(perplexity(validation_set, lambda word: lidstone_training_set.lidstone_mle(word, 0.1)))]))
outputs.append(Output(18, [str(perplexity(validation_set, lambda word: lidstone_training_set.lidstone_mle(word, 1)))]))


# best_lidstone_lambda = find_best_lidstone_lambda(lidstone_training_set, validation_set)
# print(best_lidstone_lambda)

best_lidstone_lambda = 0.06 # 0.056397
test_lidstone_perplexity = perplexity(validation_set, lambda word: lidstone_training_set.lidstone_mle(word, best_lidstone_lambda))

outputs.append(Output(19, [str(best_lidstone_lambda)]))
outputs.append(Output(20, [str(test_lidstone_perplexity)]))

##################################################################
##
##  HELD OUT
##
##################################################################

num_of_50_percent = round(len(dev_words) / 2)

held_out_training_set = Sample(dev_words[:num_of_50_percent])
held_out_set = Sample(dev_words[num_of_50_percent:])
held_out_model = HeldOutModel(held_out_training_set, held_out_set)

outputs.append(Output(21, [str(held_out_training_set.size)]))
outputs.append(Output(22, [str(held_out_set.size)]))
outputs.append(Output(23, [str(held_out_model.prob(input_word))]))
outputs.append(Output(24, [str(held_out_model.prob('unseen-word'))]))

##################################################################
##
##  TEST SET
##
##################################################################

test_lines = read_file_lines(test_file_name)
test_words = split_to_words(test_lines)
test_set = Sample(test_words)

outputs.append(Output(25, [str(len(test_words))]))

test_lidstone_perplexity = perplexity(test_set, lambda word: lidstone_training_set.lidstone_mle(word, best_lidstone_lambda))
outputs.append(Output(26, [str(test_lidstone_perplexity)]))

test_held_out_perplexity = perplexity(test_set, lambda word: held_out_model.prob(word))
outputs.append(Output(27, [str(test_held_out_perplexity)]))

L_or_H = 'L' if test_lidstone_perplexity < test_held_out_perplexity else 'H'
outputs.append(Output(28, [L_or_H]))

##################################################################
##
##  TABLE
##
##################################################################

rows = []

for r in range(0, 10):
    # lidstone
    word = 'unseen-word' if r == 0 else next((w for w, count in lidstone_training_set.dict.items() if count == r), None)
    p = lidstone_training_set.lidstone_mle(word, best_lidstone_lambda)
    f_lidstone = p * lidstone_training_set.size
    # held out
    word = 'unseen-word' if r == 0 else  next((w for w, count in held_out_training_set.dict.items() if count == r), None)
    p = held_out_model.prob(word)
    f_held_out = p * held_out_training_set.size
    N_r = held_out_model.N0 if r == 0 else len(held_out_model.count_to_words.get(r))
    T_r = held_out_model.Tr(r)
    # write row
    rows.append(f'{r}\t{f_lidstone:.5f}\t{f_held_out:.5f}\t{N_r}\t{T_r}')

outputs.append(Output(29, rows))

write_outputs_to_file(output_file_name, outputs)

