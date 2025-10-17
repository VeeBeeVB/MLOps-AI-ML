def find_duplicates(s):
    duplicates = []
    length = len(s)

    for i in range(length):
        count = 1
        for j in range(i + 1, length):
            if s[i] == s[j] and s[i] not in duplicates:
                count += 1
                print(count)
        if count > 1 and s[i] not in duplicates:
            duplicates.append(s[i])

    return duplicates

# Example usage
string = "Vijayabaskar"
print("Duplicate characters:", find_duplicates(string))
