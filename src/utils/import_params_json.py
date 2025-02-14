import json

def load_config(json_file, categories=None):
    """
    Loads a JSON file and returns a dictionary of the specified categories.
    Strings with spaces are converted to lists.
    
    Parameters:
        json_file (str): Path to the JSON file.
        categories (list, optional): List of categories to import. If None, all categories are imported.
    
    Returns:
        dict: A dictionary containing the loaded variables.
    """
    # Load the JSON file
    with open(json_file, 'r') as file:
        config = json.load(file)

    # Function to recursively process data
    def process_data(data):
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = process_data(value)
            else:
                # Convert string values to lists if they contain spaces
                if isinstance(value, str) and ' ' in value:
                    value = value.split()
                result[key] = value
        return result

    # If categories are specified, only load those
    if categories:
        result = {}
        for category in categories:
            if category in config:
                result[category] = process_data(config[category])
        return result
    else:
        # Load all categories if none are specified
        return process_data(config)