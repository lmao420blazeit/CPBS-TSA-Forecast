def format_and_capitalize(input_string):
    # Replace underscores with spaces
    modified_string = input_string.replace("_", " ")

    # Capitalize the first letter of every word
    modified_string = modified_string.title()

    return modified_string