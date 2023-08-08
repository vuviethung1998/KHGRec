import time
from SELFRec import SELFRec
from util.conf import ModelConf, namespace_to_dict
import argparse

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add the arguments
    parser.add_argument('--model', type=str, default='KGAT',
                        help='Model name')
    parser.add_argument('--dataset', type=str, default='lastfm', choices=['lastfm', 'amazon_books', 'ml-1m'],
                        help='Dataset name')
    parser.add_argument('--seed', type=int, default=123,
                        help='seed')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='KG loss pct')
    parser.add_argument('--lrate', type=float, default=0.01,
                        help='Lrate')
    parser.add_argument('--lratekg', type=float, default=0.001,
                        help='Lrate KG')
    parser.add_argument('--max_epoch', type=int, default=1,
                        help='Max Epoch')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--batch_size_kg', type=int, default=8192,
                        help='Batch size KG')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='n_layers')
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Embedding size')
    parser.add_argument('--input_dim', type=int, default=32,
                        help='Input dim')
    parser.add_argument('--relation_dim', type=int, default=32,
                        help='relation_dim')
    parser.add_argument('--hyper_dim', type=int, default=128,
                        help='hyper_dim')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='lr_decay')
    parser.add_argument('--laplacian_type', type=str, default='random-walk',
                        help='Specify the type of the adjacency (laplacian) matrix from {symmetric, random-walk}.')
    parser.add_argument('--aggregation_type', type=str, default='bi-interaction',
                        help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--reg', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--reg_kg', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')

    parser.add_argument('--cf_print_every', type=int, default=1,
                        help='Iter interval of printing CF loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=10,
                        help='Epoch interval of evaluating CF.')

    # Parse the arguments
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # Register your model here
    graph_baselines = ['LightGCN','DirectAU','MF','SASRec', 'KGAT']
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

    dict_args = namespace_to_dict(args)

    s = time.time()
    if model in graph_baselines or model in ssl_graph_models or model in sequential_baselines or model in ssl_sequential_models:
        conf = ModelConf('./conf/' + model + '.conf')
        conf.config['dataset'] = dataset
    else:
        print('Wrong model name!')
        exit(-1)
    rec = SELFRec(conf, dict_args)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
