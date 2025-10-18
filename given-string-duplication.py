def find_duplicates(s):
    duplicates = {}
    for char in s:
        if char in duplicates:
            duplicates[char] += 1
        else:
            duplicates[char] = 1

    # Filter characters that appear more than once
    dup_chars = [char for char, count in duplicates.items() if count > 1]
    return dup_chars

# Example usage
string = "programming"
print("Duplicate characters:", find_duplicates(string))
