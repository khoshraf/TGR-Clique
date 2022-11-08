import pandas as pd
import sys 
import numpy as np
import time
import random


def get_node_records_walk_no_clq(node, cut_time, clique_list, adj_list, n_walk, len_walk): 
    node_orig = node
    cut_time_orig = cut_time 
   
    walks = []
    t_walks = [] 
    eidx_walks = []
    
    for n in range(n_walk):
        node = node_orig 
        cut_time = cut_time_orig
        #print(f'node:{node}')
        #print(f'cut_time:{cut_time}')

        walk = []
        t_walk = []
        eidx_walk = []
        walk.append(node)
        t_walk.append(cut_time)
        eidx_walk.append(0)


        #len_walk-1 as the node itself is the first node in the walk
        for j in range(len_walk-1):            
            nghbrs_all = adj_list[node]
            nghbrs_time = nghbrs_all[nghbrs_all[:,2] <cut_time] 
            

            if len(nghbrs_time) > 0:
                rand = np.random.randint(0,len(nghbrs_time))

                node = nghbrs_time[rand][0]
                eid = nghbrs_time[rand][1]
                cut_time = nghbrs_time[rand][2]     
                walk.append(node)
                t_walk.append(cut_time)
                eidx_walk.append(eid)
            

            else:
                #this is where there is no neighbor so just pad it
                walk.append(0)
                t_walk.append(0)
                eidx_walk.append(0)
            

       
        walks.append(walk)
        t_walks.append(t_walk)
        eidx_walks.append(eidx_walk)

        

    
    
    return walks, t_walks, eidx_walks


#get temporal walk from cliques of a node, take one walk from all cliques and repeat until it reaches n_walk 
def get_node_records_walk_clq(node, cut_time, clique_df, adj_list,clq_adj_list, n_walk, len_walk): 

    node_orig = node
    cut_time_orig = cut_time 
    df_node = clique_df[node_orig]
    len_df_node = len(df_node)

    
    if len_df_node > 0:
        if len_df_node < n_walk:
            i_s = np.random.randint(0, len_df_node, n_walk)
        else:
            i_s = random.sample(range(0, len_df_node), n_walk)

    
    walks = []
    t_walks = [] 
    eidx_walks = []
    
    for n in range(n_walk):
        node = node_orig 
        cut_time = cut_time_orig
        
        if len(df_node) > 0:
            clq_index = df_node[i_s[n]]
            clq = clq_adj_list[clq_index]
            #print('clq',clq)
        else:
            clq =np.array([[node,0,cut_time]])
            #print('empty_clq',clq) 


        walk = []
        t_walk = []
        eidx_walk = []
        walk.append(node)
        t_walk.append(cut_time)
        eidx_walk.append(0)


        #len_walk-1 as the node itself is the first node in the walk
        for j in range(len_walk-1):#get 1 walk from 1 clique
            nghbrs_clq_time = clq[(clq[:,2]< cut_time) &((clq[:,0]==node) | (clq[:,1]==node))]

            if len(nghbrs_clq_time) > 0:
                rand = int(random.random()*len(nghbrs_clq_time))
                if nghbrs_clq_time[rand][0] == node:
                    node = nghbrs_clq_time[rand][1]
                else:
                    node = nghbrs_clq_time[rand][0]
                
                eid = nghbrs_clq_time[rand][3]
                cut_time = nghbrs_clq_time[rand][2]     
                walk.append(node)
                t_walk.append(cut_time)
                eidx_walk.append(eid)
            

            else:
                #this is where there is no neighbor so just pad it
                padding = len_walk-len(walk)
                walk += [0] * (padding)
                t_walk += [0] * (padding)
                eidx_walk += [0] * (padding)
                break
            
        
                               
        walks.append(walk)
        t_walks.append(t_walk)
        eidx_walks.append(eidx_walk)

    
    return walks, t_walks, eidx_walks

def get_node_l_records_walk(node_l, cut_time_l, clique_list, adj_list, clq_adj_list, test, n_walk, len_walk):

    len_walk = len_walk + 1 # as this is number of nodes in the walk which is one more than length of the walk
    node_l_records = np.zeros((len(node_l),n_walk, len_walk)).astype(np.int32)
    t_l_records = np.zeros((len(node_l),n_walk, len_walk)).astype(np.float32)
    eidx_l_records = np.zeros((len(node_l),n_walk, len_walk)).astype(np.int32)

    for i in range(0, len(node_l)): 
        node = node_l[i]
        #print(f'node{node}')
        cut_time = cut_time_l[i]

        if not test:
            node_records, t_records, eidx_records = get_node_records_walk_clq(node, cut_time, clique_list, adj_list,clq_adj_list, n_walk, len_walk)
        if test:
            node_records, t_records, eidx_records = get_node_records_walk_no_clq(node, cut_time, clique_list, adj_list, n_walk, len_walk)
       
        node_l_records[i] = node_records
        t_l_records[i] = t_records
        eidx_l_records[i] = eidx_records
        
        
    return node_l_records, t_l_records, eidx_l_records



