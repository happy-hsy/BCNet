import sys, os


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

path = os.path.join(os.getcwd(), 'lib')
add_path(path)
print ('(_init_paths.py) => add path to sys: {}'.format(path))

path = os.path.join(os.getcwd(), 'run')
add_path(path)
print ('(_init_paths.py) => add path to sys: {}'.format(path))


