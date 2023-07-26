import time
from SELFRec import SELFRec
from util.conf import ModelConf
import argparse

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add the arguments
    parser.add_argument('--model', type=str, default='HGNN',
                        help='Model name')
    parser.add_argument('--dataset', type=str, default='lastfm',
                        help='Dataset name')

    # Parse the arguments
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # Register your model here
    graph_baselines = ['LightGCN','DirectAU','MF','SASRec']
    ssl_graph_models = ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL','MixGCF', 'HKGRippleNet', 'HGNN', 'HCCF']
    sequential_baselines= ['SASRec']
    ssl_sequential_models = ['CL4SRec']

    args = parse_arguments()

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print('=' * 80)

    print('Graph-Based Baseline Models:')
    print('   '.join(graph_baselines))
    print('-' * 100)
    print('Self-Supervised  Graph-Based Models:')
    print('   '.join(ssl_graph_models))
    print('=' * 80)
    print('Sequential Baseline Models:')
    print('   '.join(sequential_baselines))
    print('-' * 100)
    print('Self-Supervised Sequential Models:')
    print('   '.join(ssl_sequential_models))
    print('=' * 80)
    # model = input('Please enter the model you want to run:')
    # model ='MHCN'
    # model  = 'HKGRippleNet'
    # model = 'SSL4Rec'
    model = args.model
    dataset = args.dataset

    s = time.time()
    if model in graph_baselines or model in ssl_graph_models or model in sequential_baselines or model in ssl_sequential_models:
        conf = ModelConf('./conf/' + model + '.conf')
        conf.config['dataset'] = dataset
    else:
        print('Wrong model name!')
        exit(-1)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
