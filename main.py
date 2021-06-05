import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import torch.optim as optim
import csv
import os
import sys
import datetime
import math
from Models import *

from utility.helper import *
from utility.batch_test import *


class Model_Wrapper(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = args.model_type
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.mess_dropout = eval(args.mess_dropout)

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']



        self.norm_adj = data_config['norm_adj']
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float()


        self.record_alphas = False

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)


        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        print('model_type is {}'.format(self.model_type))

        self.weights_save_path = '%sweights/%s/%s/l%s/r%s' % (args.weights_path, args.dataset, self.model_type,
                                                                 str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        """

        print('----self.alg_type is {}----'.format(self.alg_type))

        if self.alg_type in ['ngcf']:
            self.model = NGCF(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout)
        elif self.alg_type in ['mf']:
            self.model = MF(self.n_users, self.n_items, self.emb_dim)
        else:
            raise Exception('Dont know which model to train')

        self.model = self.model.cuda()
        self.norm_adj = self.norm_adj.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.lr_scheduler = self.set_lr_scheduler()
        print(self.model)
        for name, param in self.model.named_parameters():
            print(name, ' ', param.size())

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def save_model(self):
        ensureDir(self.weights_save_path)
        torch.save(self.model.state_dict(), self.weights_save_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.weights_save_path))

    def test(self, users_to_test, drop_flag=False, batch_test_flag=False):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.norm_adj)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test)
        return result

    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, map_loger, mrr_loger, fone_loger = [], [], [], [], [], [], [], []
        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0.

        n_batch = data_generator.n_train // args.batch_size + 1

        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            sample_time = 0.
            cuda_time = 0.
            for idx in range(n_batch):
                self.model.train()
                self.optimizer.zero_grad()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1

                ua_embeddings, ia_embeddings = self.model(self.norm_adj)

                u_g_embeddings = ua_embeddings[users]
                pos_i_g_embeddings = ia_embeddings[pos_items]
                neg_i_g_embeddings = ia_embeddings[neg_items]

                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)

                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                batch_loss.backward()
                self.optimizer.step()

                loss += float(batch_loss)
                # print('loss: ', loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)


            self.lr_scheduler.step()

            del ua_embeddings, ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings

            if math.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
            if (epoch + 1) % 10 != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                        epoch, time() - t1, loss, mf_loss, emb_loss)
                    #training_time_list.append(time() - t1)
                    print(perf_str)
                continue

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            ret = self.test(users_to_test, drop_flag=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            map_loger.append(ret['map'])
            mrr_loger.append(ret['mrr'])
            fone_loger.append(ret['fone'])
            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f], map=[%.5f, %.5f], mrr=[%.5f, %.5f], f1=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1],ret['map'][0], ret['map'][-1],ret['mrr'][0], ret['mrr'][-1],ret['fone'][0], ret['fone'][-1])
                print(perf_str)
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=5)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop:
                break

            # *********************************************************
            # save the user & item embeddings for pretraining.
            if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
                # save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
                self.save_model()
                if self.record_alphas:
                    self.best_alphas = [i for i in self.model.get_alphas()]
                print('save the weights in path: ', self.weights_save_path)

        #print the final recommendation results to csv files.
        if args.save_recom:
            results_save_path = '%soutput/%s/reRecommendation.csv' % (args.proj_path, args.dataset)
            self.save_recResult(results_save_path)
        
        if rec_loger != []:
            self.print_final_results(rec_loger, pre_loger, ndcg_loger, hit_loger, map_loger,mrr_loger,fone_loger,training_time_list)

    def save_recResult(self,outputPath):
        #used for reverve the recommendation lists
        recommendResult = {}
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE
        
        #get all apps (users)
        users_to_test = list(data_generator.test_set.keys())
        n_test_users = len(users_to_test)
        n_user_batchs = n_test_users // u_batch_size + 1
        count = 0
        
        #calculate the result by our own
        #get the latent factors
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.norm_adj)
            
        #get result in batch
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = users_to_test[start: end]
            item_batch = range(ITEM_NUM)
            u_g_embeddings = ua_embeddings[user_batch]
            i_g_embeddings = ia_embeddings[item_batch]
            #get the ratings
            rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))
            #move from GPU to CPU
            rate_batch = rate_batch.detach().cpu().numpy()
            #contact each user's ratings with his id
            user_rating_uid = zip(rate_batch, user_batch)
            #now for each user, calculate his ratings and recommendation
            for x in user_rating_uid:
                #user u's ratings for user u
                rating = x[0]
                #uid
                u = x[1]
                training_items = data_generator.train_items[u]
                user_pos_test = data_generator.test_set[u]
                all_items = set(range(ITEM_NUM))
                test_items = list(all_items - set(training_items))
                item_score = {}
                for i in test_items:
                    item_score[i] = rating[i]
                K_max = max(Ks)
                K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
                recommendResult[u] = K_max_item_score
     
        
        #output the result to csv file.
        ensureDir(outputPath)
        with open(outputPath, 'w') as f:
            print("----the recommend result has %s items." % (len(recommendResult)))
            for key in recommendResult.keys(): #due to that all users have been used for test and the subscripts start from 0.
                outString = ""
                for v in recommendResult[key]:
                    outString = outString + "," + str(v)
                f.write("%s%s\n"%(key,outString))
                #f.write("%s,%s\n"%(key,recommendResult[key]))
    
    
    def print_final_results(self, rec_loger, pre_loger, ndcg_loger, hit_loger,map_loger,mrr_loger,fone_loger,training_time_list):
        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        map = np.array(map_loger)
        mrr = np.array(mrr_loger)
        fone = np.array(fone_loger)

        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s], map=[%s],mrr=[%s], f1=[%s]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in pres[idx]]),
                      '\t'.join(['%.5f' % r for r in map[idx]]),
                      '\t'.join(['%.5f' % r for r in mrr[idx]]),
                      '\t'.join(['%.5f' % r for r in fone[idx]]))
                      
        #output the result that can be uesed by Matlab.
        final_perf_output = "%s\n%s\n%s\n%s\n%s\n%s\n%s" % \
                     (','.join(['%.5f' % r for r in recs[idx]]),
                      ','.join(['%.5f' % r for r in pres[idx]]),
                      ','.join(['%.5f' % r for r in map[idx]]),
                      ','.join(['%.5f' % r for r in mrr[idx]]),
                      ','.join(['%.5f' % r for r in fone[idx]]))
        print(final_perf)

        # Benchmarking: time consuming
        avg_time = sum(training_time_list) / len(training_time_list)
        time_consume = "Benchmarking time consuming: average {}s per epoch".format(avg_time)
        print(time_consume)

        results_path = '%soutput/%s/result.csv' % (args.proj_path, args.dataset)

        ensureDir(results_path)
        f = open(results_path, 'a')

        f.write(final_perf_output)
        f.close()


    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def   sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_sparse_tensor_value(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape


def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    if args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')

    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    Engine = Model_Wrapper(data_config=config, pretrain_data=pretrain_data)
    if args.pretrain:
        print('pretrain path: ', Engine.weights_save_path)
        if os.path.exists(Engine.weights_save_path):
            Engine.load_model()
            users_to_test = list(data_generator.test_set.keys())
            ret = Engine.test(users_to_test, drop_flag=True)
            cur_best_pre_0 = ret['recall'][0]

            pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                           'ndcg=[%.5f, %.5f], map=[%.5f, %.5f], mrr=[%.5f, %.5f], f1=[%.5f, %.5f]' % \
                           (ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1],
                            ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1],
                            ret['map'][0], ret['map'][-1],
                            ret['mrr'][0], ret['mrr'][-1],
                            ret['fone'][0], ret['fone'][-1])
            print(pretrain_ret)
        else:
            print('Cannot load pretrained model. Start training from stratch')
    else:
        print('without pretraining')
    Engine.train()