import numpy as np
import pandas as pd
import os
import math
from collections import defaultdict

class homework1:

    file_name = os.getcwd()+ '/' + 'hw1_1_dat.txt'   
    
    def __init__(self):
        self.hw_1a, self.hw_1b = self.hw_1ab_fun(self.file_name)

    def hw_1ab_fun(self, file_name):
        answer_1a={}
        answer_1b=pd.DataFrame()
        with open(file_name, 'r') as file:
            lines=file.readlines()
            n_lines=len(lines)
            iterations=[]
            values= []
            epochs=[]
            Ubeta2sum=[]
            log10Ubeta2sum=[]
            res_dict = defaultdict(list)
            for line in lines:
                if line.startswith("it"):
                    words = line.split(" ")
                    iteration = int(words[1])
                    iteration_a= words[1]
                    iterations.append(iteration)
                    values.append(int(words[4]))
                    epochs.append(int(words[7]))
                    Ubeta2sum.append(float(words[13]))
                    val = int(words[4])
                    if iteration_a not in res_dict.keys():
                        res_dict[iteration_a].append(val)
                    else:
                        if val not in res_dict[iteration_a]:
                            res_dict[iteration_a].append(val)
                        
            answer_1a = {"n_lines":n_lines,"n_it":len(set(iterations)), "it": res_dict}
            log10Ubeta2sum=[math.log(x,10) for x in Ubeta2sum]
            df = pd.DataFrame({'it':iterations,
                                'v':values,
                                'epoch':epochs,
                                'log10Ubeta2sum':log10Ubeta2sum})
            answer_1b=df
        return answer_1a, answer_1b
       
    def findmin(self, it, v, df):
        filtered_df = df.loc[(df.it==it) & (df.v==v)]
        smallest = filtered_df.log10Ubeta2sum.min()
        return smallest
