from collections import Counter

def find_duplicates(s):
    counter = Counter(s)
    return [char for char, count in counter.items() if count > 1]

# Example usage
string = "programming"
print("Duplicate characters:", find_duplicates(string))
