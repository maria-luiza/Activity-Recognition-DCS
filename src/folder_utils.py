import os


def get_root_dirname():
    """ Return the project root directory name """
    return os.path.dirname(os.path.dirname(__file__))


def join_paths(dir, path):
    """ Return the joined path with the root """
    return os.path.join(dir, path)


def list_directory(path):
    """ List all the directories inside a path"""
    return os.listdir(path)


def open_dataset_files(dir):
    """
    Return the dataset as list of data events
    :param dir:
    :return:
    """
    file = open(dir, "r")
    fileRows = []
    for line in file:
        terms = line.split()
        fileRow = []
        for term in terms:
            fileRow.append(term)
        fileRows.append(fileRow)
    return fileRows


if __name__ == '__main__':
    # Get path from files
    ROOT_DIR = get_root_dirname()
    FILES_DIR = join_paths(ROOT_DIR, "datasets")

    # Get datasets on directory
    datasets = list_directory(FILES_DIR)

    for data in datasets:
        if not data.startswith('.'):
            # Get the annotated text from each dataset
            file = open_dataset_files(join_paths(FILES_DIR, data + "/ann.txt"))
            print(file)
            print("/n")
