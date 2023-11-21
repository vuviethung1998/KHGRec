import time
import sys
import os
from os.path import abspath
import torch 
import numpy as np 
import pandas as pd
import csv 

from base.recommender import Recommender
from data.ui_graph import Interaction
from data.knowledge import Knowledge
from util.algorithm import find_k_largest
from time import strftime, localtime
from data.loader import FileIO
from util.evaluation import ranking_evaluation

class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, knowledge_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        if self.knowledge:
            self.data_kg = Knowledge(conf, training_set, test_set, knowledge_set)
        self.bestPerformance = []
        top = self.ranking.split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)  
        exp = kwargs['experiment']

        if exp == 'cold_start':
            exp_name = f"cold_start_{kwargs['group_id']}"
        elif exp == 'missing':
            exp_name = f"missing_{kwargs['missing_pct']}"
        elif exp == 'add_noise':
            exp_name = f"add_noise_{kwargs['noise_pct']}"
        else:
            exp_name = 'full'
        if conf['model.name'] == 'HGNN':
            self.output = f"./results/{self.model_name}/{self.dataset}/{exp_name}/@{self.model_name}-bs:{kwargs['batch_size']}-lr:{kwargs['lrate']}-lrd:{kwargs['lr_decay']}-inp_emb:{kwargs['input_dim']}-hyper_emb:{kwargs['hyper_dim']}-reg:{kwargs['reg']}-leaky:{kwargs['p']}-dropout:{kwargs['drop_rate']}-n_layers:{kwargs['n_layers']}-temp:{kwargs['temp']}-cl_rate:{kwargs['cl_rate']}/"
        else:
            self.output = f"./results/{self.model_name}/{self.dataset}/{exp_name}/@{self.model_name}-bs:{kwargs['batch_size']}-lr:{kwargs['lrate']}-lrd:{kwargs['lr_decay']}-wdecay:{kwargs['weight_decay']}-reg:{kwargs['reg']}-leaky:{kwargs['p']}-dropout:{kwargs['drop_rate']}-n_layers:{kwargs['n_layers']}-temp:{kwargs['temp']}-cl_rate:{kwargs['cl_rate']}"
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # # print dataset statistics
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.training_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            # s_find_candidates = time.time()
            candidates = self.predict(user)
            # e_find_candidates = time.time()
            # print("Calculate candidates time: %f s" % (e_find_candidates - s_find_candidates))
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, wli = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            
            # s_find_k_largest = time.time()
            ids, scores = find_k_largest(self.max_N, candidates)
            # e_find_k_largest = time.time()
            # print("Find k largest candidates: %f s" % (e_find_k_largest - s_find_k_largest))
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for user in self.data.test_set:
            line = str(user) + ':'
            for item in rec_list[user]:
                line += ' (' + str(item[0]) + ',' + str(item[1]) + ')'
                if item[0] in self.data.test_set[user]:
                    line += '*'
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time.time()))
        # output prediction result
        out_dir = self.output + '/'
        # out_dir = self.output['-dir']
        # file_name = self.config['model.name'] + '@' + current_time + '-top-' + str(self.max_N) + 'items' + '.txt'
        file_name = self.config['model.name'] + '-top-' + str(self.max_N) + 'items' + '.txt'
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        # file_name = self.config['model.name'] + '@' + current_time + '-performance' + '.txt'
        file_name = self.config['model.name'] + '-performance' + '.txt'
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))

    def fast_evaluation(self, epoch, kwargs=None, train_time=None):
        print('Evaluating the model...')
        s_test = time.time()
        rec_list = self.test()
        e_test = time.time() 
        print("Test time: %f s" % (e_test - s_test))
        
        s_measure = time.time()
        all_measures = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        len_measures = len(self.topN)
        lst_res = []
    
        data_ep = {}
        data_ep['epoch'] = epoch
        data_ep['train_time'] = train_time
        data_ep['test_time'] = (e_test - s_test)
        for i in range(0, len(all_measures), 5):
            mes = all_measures[i:i+5]
            topk = int(mes[0].split(' ')[1][:-1])
            hitk = float(mes[1].split(':')[1][:-1])
            preck = float(mes[2].split(':')[1][:-1])
            reck = float(mes[3].split(':')[1][:-1])
            ndcgk = float(mes[4].split(':')[1][:-1])
            data_ep[f'hit@{topk}'] = hitk
            data_ep[f'precision@{topk}'] = preck
            data_ep[f'recall@{topk}'] = reck
            data_ep[f'ndcg@{topk}'] = ndcgk

        measure = all_measures[(len_measures - 1) * 5: (len_measures - 1) * 5 + 5]
        e_measure = time.time()
        print("Measure time: %f s" % (e_measure - s_measure))
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()

        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure, data_ep

    def save_model(self, model):
        # save model 
        current_time = strftime("%Y-%m-%d", localtime(time.time()))
        out_dir = self.output
        file_name =  self.config['model.name'] + '@' + current_time + '-weight' + '.pth'
        weight_file = out_dir + '/' + file_name 
        torch.save(model.state_dict(), weight_file)

    def save_loss(self, train_losses, rec_losses, reg_losses, cl_losses=None):
        df_train_loss = pd.DataFrame(train_losses, columns = ['ep', 'loss'])
        df_rec_loss = pd.DataFrame(rec_losses, columns = ['ep', 'loss'])
        df_reg_loss = pd.DataFrame(reg_losses, columns = ['ep', 'loss'])

        df_train_loss.to_csv(self.output + '/train_loss.csv')
        df_rec_loss.to_csv(self.output + '/rec_loss.csv')
        df_reg_loss.to_csv(self.output + '/reg_loss.csv')

        if not cl_losses:
            df_cl_loss = pd.DataFrame(cl_losses, columns= ['ep', 'loss'])
            df_cl_loss.to_csv(self.output + '/cl_loss.csv')

    def save_perfomance_training(self, log_train):
        df_train_log = pd.DataFrame(log_train)
        df_train_log.to_csv(self.output + '/performance.csv')

    def save_performance_row(self, ep, data_ep):
        # opening the csv file in 'w' mode
        csv_path = self.output + 'train_performance.csv'
        
        # 'Hit Ratio:0.00328', 'Precision:0.00202', 'Recall:0.00337', 'NDCG:0.00292
        hit = float(data_ep[0].split(':')[1])
        precision = float(data_ep[1].split(':')[1])
        recall = float(data_ep[2].split(':')[1])
        ndcg = float(data_ep[3].split(':')[1])

        with open(csv_path, 'a+', newline = '') as f:
            header = ['ep', 'training_time', 'testing_time','hit@20', 'prec@20', 'recall@20', 'ndcg@20']
            writer = csv.DictWriter(f, fieldnames = header)
            # writer.writeheader()
            writer.writerow({
                 'ep' : ep,
                 'training_time': train_time,
                 'testing_time': test_time, 
                 'hit@20': hit,
                 'prec@20': precision,
                 'recall@20': recall,
                 'ndcg@20': ndcg,
            })