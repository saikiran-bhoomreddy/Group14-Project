import radon
from radon.visitors import ComplexityVisitor
from radon.complexity import cc_rank

def calculate_average_complexity(file_path):
    with open(file_path, 'r') as file:
        code = file.read()

    visitor = ComplexityVisitor.from_code(code)
    complexity_list = [block.complexity for block in visitor.blocks]

    if complexity_list:
        average_complexity = sum(complexity_list) / len(complexity_list)
    else:
        average_complexity = 0

    return average_complexity

def print_complexity_scale(average_complexity):
    scale = cc_rank(average_complexity)

    print(f"Average Complexity: {average_complexity:.2f}")
    print("Complexity Scale: {}".format(scale))

if __name__ == "__main__":
    file_path = './1.py'

    average_complexity = calculate_average_complexity(file_path)
    print_complexity_scale(average_complexity)
