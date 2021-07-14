import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import loadmat
import os
import re
from glob import glob

def load_hrir(path):
    names = []
    names += glob(path)
    print(names[0])

    splitnames = [os.path.split(name) for name in names]
    print(len(splitnames))

    p = re.compile('IRC_\d{4,4}')
    print(p)

    subjects = [int(name[4:8]) for base, name in splitnames 
                            if not (p.match(name[-8:]) is None)]
    print(subjects)

    k = 0
    subject = subjects[k]

    for k in range(len(names)):
        subject = subjects[k]
        # filename = os.path.join(names[k], 'IRC_' + str(subject))
        filename = os.path.join(names[k], 'COMPENSATED/MAT/HRIR/IRC_' + str(subject) + '_C_HRIR.mat')
    #     print(filename)

    m = loadmat(filename, struct_as_record=True)
    print(m.keys())
    print(m['l_eq_hrir_S'].dtype)

    l, r = m['l_eq_hrir_S'], m['r_eq_hrir_S']
    hrirSet_l = l['content_m'][0][0]
    hrirSet_r = r['content_m'][0][0]
    elev = l['elev_v'][0][0]
    azim = l['azim_v'][0][0]
    fs_HRIR = m['l_eq_hrir_S']['sampling_hz'][0][0][0][0]

    locLabel = np.hstack((elev, azim))
    print("locLabel shape: ", locLabel.shape, " (order: elev, azim)")
    # print(locLabel[0:5])

    # 0: left-ear 1: right-ear
    hrirSet = np.vstack((np.reshape(hrirSet_l, (1,) + hrirSet_l.shape),
                            np.reshape(hrirSet_r, (1,) + hrirSet_r.shape)))
    hrirSet = np.transpose(hrirSet, (1,0,2))
    print("hrirSet shape: ", hrirSet.shape)

    return hrirSet, locLabel, fs_HRIR



if __name__ == "__main__":
    path = "./HRTF/IRC*"
    hrirSet, locLabel, fs_HRIR = load_hrir(path)