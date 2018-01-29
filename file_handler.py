import os

def read_and_parse_dir(src_path):
    '''Recursively read all filenames in dirs and subdirs
    Return full path files list, in which each sublist
    includes file path in that directory.
    '''
    samples = []
    for root, dirs, files in os.walk(src_path):
        for name in sorted(files):
            samples.append(os.path.join(root, name))
    samples_list = []
    dir_name = None
    dir_samples = []
    for sample in samples:
        name = sample.split('/')[-2]
        if not dir_name == name:
            if len(dir_samples) > 0:
                samples_list.append(dir_samples)
            dir_samples = [sample]
            dir_name = name
        else:
            dir_samples.append(sample)
    samples_list.append(dir_samples)
    return samples_list
