import random
import numpy as np 

if __name__=="__main__":
    train = []
    test = []
    with open('ratings.dat') as f:
        for line in f:
            items = line.strip().split('::')
            new_line = '\t'.join(items[:-2])+'\t1\n'
            if int(items[-2]) <4:
                continue
            if random.random() > 0.1:
                train.append(new_line)
            else:
                test.append(new_line)
    with open('train.txt','w') as f:
        f.writelines(train)

    with open('test.txt','w') as f:
        f.writelines(test)