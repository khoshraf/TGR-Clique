import pandas as pd
from log import *
from parser import *
from eval import *
from utils import *
from train import *
import numba
from module import TGR_Clique
import resource
import time
import records

args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
GPU = args.gpu
USE_TIME = args.time
ATTN_AGG_METHOD = args.attn_agg_method
ATTN_MODE = args.attn_mode
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
WALK_POOL = args.walk_pool
WALK_N_HEAD = args.walk_n_head
TOLERANCE = args.tolerance
CPU_CORES = args.cpu_cores
NGH_CACHE = args.ngh_cache
VERBOSITY = args.verbosity
AGG = args.agg
SEED = args.seed
clique_file = args.clq_file
NUM_WALK = args.num_walk #should be changed according to dataset
LEN_WALK = args.len_walk #should be changed according to dataset

assert(CPU_CORES >= -1)
set_random_seed(SEED)
logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)

# Load data and sanity check
g_df = pd.read_csv('data/ml_{}.csv'.format(DATA))
e_feat = np.load('data/ml_{}.npy'.format(DATA))
n_feat = np.load('data/ml_{}_node.npy'.format(DATA))

#This is just to get length of the longest line
clique_df = pd.read_csv(clique_file, skiprows=2, header=None)
clique_df['len'] = clique_df[0].apply(lambda x: len(x.split(' ')))
max_len = max(clique_df['len'])
#print('max_len:', max_len)
del clique_df

#This is reading the clique_df
clique_df = pd.read_csv(clique_file, skiprows=2, header=None, delimiter = ' ', names = [a for a in range(0, max_len)])
clique_df = clique_df.apply(lambda x: x+1)
clique_df = clique_df.fillna(0)

#print('clique_df',clique_df)


src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values
max_idx = max(src_l.max(), dst_l.max())
#print('max_idx', max_idx)
#print('2', np.unique(np.stack([src_l, dst_l])).shape[0])

assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx)  # all nodes except node 0 should appear and be compactly indexed
assert(n_feat.shape[0] == max_idx + 1)  # the nodes need to map one-to-one to the node feat matrix




# split and pack the data by generating valid train/val/test mask according to the "mode"
val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
valid_train_flag2 = (ts_l < val_time)

if args.mode == 't':
    logger.info('Transductive training...')
    #####I change here and half the len
    #valid_train_flag = (ts_l <= val_time)
    valid_train_flag = (ts_l <= val_time)
    
    
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time

else:
    assert(args.mode == 'i')
    logger.info('Inductive training...')
    # pick some nodes to mask (i.e. reserved for testing) for inductive setting
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)

    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (none_mask_node_flag > 0.5)  # both train and val edges can not contain any masked nodes
    valid_test_flag = (ts_l > test_time) * (none_mask_node_flag < 0.5)  # test edges must contain at least one masked node
    valid_test_new_new_flag = (ts_l > test_time) * mask_src_flag * mask_dst_flag
    valid_test_new_old_flag = (valid_test_flag.astype(int) - valid_test_new_new_flag.astype(int)).astype(bool)
    logger.info('Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(len(mask_node_set)))

# split data according to the mask
train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag], label_l[valid_train_flag]
val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag], label_l[valid_val_flag]
test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag]
if args.mode == 'i':
    test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l = src_l[valid_test_new_new_flag], dst_l[valid_test_new_new_flag], ts_l[valid_test_new_new_flag], e_idx_l[valid_test_new_new_flag], label_l[valid_test_new_new_flag]
    test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l = src_l[valid_test_new_old_flag], dst_l[valid_test_new_old_flag], ts_l[valid_test_new_old_flag], e_idx_l[valid_test_new_old_flag], label_l[valid_test_new_old_flag]
train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
train_val_data = (train_data, val_data)


####################################################################
#This is for creating clique for training
max_idx_train = max(train_src_l.max(), train_dst_l.max())

clique_list = [[] for _ in range(max_idx_train + 1)]
zip_src_dst_l_train = np.unique(np.concatenate((train_src_l, train_dst_l)))


for src_dst_n in zip_src_dst_l_train:
    clique_list[src_dst_n] = clique_df.index[(clique_df == src_dst_n).any(1)].tolist()
   
    
print('1****clique_list_done')

####################################################################
#get time for edges in the cliques
train_src_l_tmp, train_dst_l_tmp, train_ts_l_tmp, train_e_idx_l_tmp, train_label_l_tmp = src_l[valid_train_flag2], dst_l[valid_train_flag2], ts_l[valid_train_flag2], e_idx_l[valid_train_flag2], label_l[valid_train_flag2]
#print('**********len',len(train_src_l_tmp))

clq_adj_list = [[] for _ in range(len(clique_df))]

for src, dst, eidx, ts in zip(train_src_l_tmp, train_dst_l_tmp, train_e_idx_l_tmp, train_ts_l_tmp):
    clq_indices = clique_df.index[(clique_df == src).any(1) & (clique_df == dst).any(1)].tolist()
           
    for i in clq_indices:
        clq_adj_list[i].append((src,dst,ts,eidx))

print('2***clq_adj_list_done')

####################################################################


full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))

#partial_adj_list = [[] for _ in range(max_idx + 1)]
#for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
#    partial_adj_list[src].append((dst, eidx, ts))
#    partial_adj_list[dst].append((src, eidx, ts))
#for src, dst, eidx, ts in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l):
#    partial_adj_list[src].append((dst, eidx, ts))
#    partial_adj_list[dst].append((src, eidx, ts))


# create random samplers to generate train/val/test instances
train_rand_sampler = RandEdgeSampler((train_src_l, ), (train_dst_l, ))
val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
rand_samplers = train_rand_sampler, val_rand_sampler

# multiprocessing memory setting
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (200*args.bs, rlimit[1]))

# model initialization
device = torch.device('cuda:{}'.format(GPU))
tgr_clique = TGR_Clique(n_feat, e_feat, agg=AGG,
            num_layers=NUM_LAYER, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
            n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, 
            walk_n_head=WALK_N_HEAD, walk_linear_out=args.walk_linear_out,
            cpu_cores=CPU_CORES, verbosity=VERBOSITY, get_checkpoint_path=get_checkpoint_path, adj_list = full_adj_list, clique_list = clique_list, clq_adj_list=clq_adj_list
            , num_walk = NUM_WALK, len_walk = LEN_WALK)
tgr_clique.to(device)
optimizer = torch.optim.Adam(tgr_clique.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

# start train and val phases

start_time = time.time()
train_val(train_val_data, tgr_clique, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, rand_samplers, logger)
print('train:----%s seconds----'% (time.time() - start_time))


# final testing
start_time2 = time.time()
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for {} nodes'.format(args.mode), tgr_clique, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l)
print('test:----%s seconds----'% (time.time() - start_time2))



logger.info('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))
test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6
#print('here')
if args.mode == 'i':
    test_new_new_acc, test_new_new_ap, test_new_new_f1, test_new_new_auc = eval_one_epoch('test for {} nodes'.format(args.mode), tgr_clique, test_rand_sampler, test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_label_new_new_l, test_e_idx_new_new_l)
    logger.info('Test statistics: {} new-new nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_new_acc, test_new_new_ap, test_new_new_auc ))
    test_new_old_acc, test_new_old_ap, test_new_old_f1, test_new_old_auc = eval_one_epoch('test for {} nodes'.format(args.mode), tgr_clique, test_rand_sampler, test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_label_new_old_l, test_e_idx_new_old_l)
    logger.info('Test statistics: {} new-old nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_old_acc, test_new_old_ap, test_new_old_auc))

# save model
logger.info('Saving TGR_Clique model ...')
torch.save(tgr_clique.state_dict(), f'./saved_models/{DATA}.path')
#torch.save(tgr_clique.state_dict(), best_model_path)



logger.info('TGR_Clique model saved')


