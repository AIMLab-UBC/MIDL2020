import numpy as np


def combine(file_a, file_b, out_path):
    with open(file_a) as f:
        data_str = f.read()
        data_str = data_str.strip()
    patch_infos_a = data_str.split('---\n')

    with open(file_b) as f:
        data_str = f.read()
        data_str = data_str.strip()
    patch_infos_b = data_str.split('---\n')

    with open(out_path, 'w') as f:
        for idx, patch_info_a in enumerate(patch_infos_a):
            patch_info_a = patch_info_a.split('\n')
            patch_info_b = patch_infos_b[idx].split('\n')
            assert(patch_info_a[0].split('/')[-1] ==
                   patch_info_b[0].split('/')[-1])
            prob_a = np.fromstring(
                patch_info_a[1][1:-1], count=5, sep=' ')
            prob_b = np.fromstring(
                patch_info_b[1][1:-1], count=5, sep=' ')
            prob_avg = (prob_a + prob_b) / 2
            f.write('{}\n'.format(patch_info_a[0]))
            f.write('{}\n'.format(str(prob_avg).replace('\n', '')))
            f.write('---\n')


for i in ['a', 'b', 'c', 'd', 'e', 'f']:
    combine('/Users/Andy/Desktop/results_maestro_v2/testing_' + i + '_distribution.txt',
            '/Users/Andy/Desktop/results_maestro_v2/testing_' + i + '_distribution_256.txt',
            '/Users/Andy/Desktop/results_maestro_v2/testing_' + i + '_distribution_avg.txt')
