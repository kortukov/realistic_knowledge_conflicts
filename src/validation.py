def ensure_string_fields(dataset, fields):
    """Ensure that all examples in the dataset have all the fields as strings

    If the field is not present, it is added with an empty string.
    
    Mutates the dataset. 
    """
    for example in dataset:
        for field in fields:
            value_to_write = str(example.get(field, ""))
            example[field] = value_to_write


def assert_fields_exist(dataset, fields):
    """Assert that all examples in the dataset have all the fields.
    
    If field is a list then any of the fields should exist.
    """
    for i, example in enumerate(dataset):
        for field in fields:
            if isinstance(field, list):
                assert any([f in example for f in field]), f"None of {field} is in example {i}: {example}"
            else:
                assert field in example, f"Field {field} not in example {i}: {example}"
