import math


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num/total_num,5)

    # # @staticmethod
    # def hit_ratio(origin, hits):
    #     """
    #     Note: This type of hit ratio calculates the fraction:
    #      (# users who are recommended items in the test set / #all the users in the test set)
    #     """
    #     hit_num = 0
    #     for user in hits:
    #         if hits[user] > 0:
    #             hit_num += 1
    #     return hit_num / len(origin)

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N),5)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list),5)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall),5)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return round(error/count,5)

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return round(math.sqrt(error/count),5)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2,2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2,2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res),5)

class Measure:
    def __init__(self):
        self.hit1  = {"raw": 0.0, "fil": 0.0}
        self.hit3  = {"raw": 0.0, "fil": 0.0}
        self.hit10 = {"raw": 0.0, "fil": 0.0}
        self.mrr   = {"raw": 0.0, "fil": 0.0}
        self.mr    = {"raw": 0.0, "fil": 0.0}

    def __add__(self, other):
        """
        Add two measure objects
        """
        new_measure = Measure()
        settings = ["raw", "fil"]

        for rf in settings:
            new_measure.hit1[rf] = (self.hit1[rf] + other.hit1[rf])
            new_measure.hit3[rf] = (self.hit3[rf] + other.hit3[rf])
            new_measure.hit10[rf] = (self.hit10[rf] + other.hit10[rf])
            new_measure.mrr[rf] = (self.mrr[rf] + other.mrr[rf])
            new_measure.mr[rf] = (self.mr[rf] + other.mr[rf])
        return new_measure

    def __str__(self):
        str_list = []
        for raw_or_fil in ["raw", "fil"]:
            str_list.append("{} setting:".format(raw_or_fil.title()))
            str_list.append("\tHit@1  = {}".format(self.hit1[raw_or_fil]))
            str_list.append("\tHit@3  = {}".format(self.hit3[raw_or_fil]))
            str_list.append("\tHit@10 = {}".format(self.hit10[raw_or_fil]))
            str_list.append("\tMR  = {}".format(self.mr[raw_or_fil]))
            str_list.append("\tMRR = {}".format(self.mrr[raw_or_fil]))
            str_list.append("")
        return '\n'.join(str_list)

    def __repr__(self):
        return self.__str__()

    def update(self, rank, raw_or_fil):
        if rank == 1:
            self.hit1[raw_or_fil] += 1.0
        if rank <= 3:
            self.hit3[raw_or_fil] += 1.0
        if rank <= 10:
            self.hit10[raw_or_fil] += 1.0

        self.mr[raw_or_fil]  += rank
        self.mrr[raw_or_fil] += (1.0 / rank)

    def normalize(self, normalizer):
        if normalizer == 0:
            return
        for raw_or_fil in ["raw", "fil"]:
            self.hit1[raw_or_fil]  /= (normalizer)
            self.hit3[raw_or_fil]  /= (normalizer)
            self.hit10[raw_or_fil] /= (normalizer)
            self.mr[raw_or_fil]    /= (normalizer)
            self.mrr[raw_or_fil]   /= (normalizer)

def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        # F1 = Metric.F1(prec, recall)
        # indicators.append('F1:' + str(F1) + '\n')
        #MAP = Measure.MAP(origin, predicted, n)
        #indicators.append('MAP:' + str(MAP) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        # AUC = Measure.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure

def rating_evaluation(res):
    measure = []
    mae = Metric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = Metric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure

def early_stopping(recall_list, stopping_steps):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop