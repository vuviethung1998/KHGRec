import argparse

from base.kggraph_recommender import KnowledgeGraphEmbedding 
from util.conf import ModelConf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default="ml-1M")  
    parser.add_argument("-test", action="store_true", help="If -test is set, then you must specify a -pretrained model. "
                        + "This will perform testing on the pretrained model and save the output in -output_dir")
    parser.add_argument("-no_test_by_arity", action="store_true", help="If set, then validation will be performed by arity.")
    parser.add_argument("-test_by_op", action="store_true", help="If set, then validation will be performed by operation.")
    parser.add_argument("-test_by_deg", action="store_true", help="If set, then validation will be performed by degree.")
    parser.add_argument("-general_test", action="store_true", help="If set, then validation will be performed for all.")
    
    parser.add_argument('-num_interations', type=int, default=1, help="Number of interations")
    parser.add_argument('-pretrained', type=str, default=None, help="A path to a trained model (.chkpnt file), which will be loaded if provided.")
    parser.add_argument('-output_dir', type=str, default=None, help="A path to the directory where the model will be saved and/or loaded from.")
    parser.add_argument('-restartable', action="store_true", help="If restartable is set, you must specify an output_dir")
    parser.add_argument('-non_linearity', type=str, default="sigmoid", help="non-linearity function to apply for each step of RealE")
    parser.add_argument('-ent_non_linearity', type=str, default="sigmoid", help="non-linearity to apply on entity embeddings")
    parser.add_argument('-smart_initialization', action="store_true")
    parser.add_argument('-opt', type=str, default="Adagrad")
    args = parser.parse_args()
    return args  

if __name__ == '__main__':
    # Register your model here
    
    kg_embedding_models = ['RealE']
    model = 'RealE'

    # print('KG HyperGraph Embedding Models:')
    # print('   '.join(kg_embedding_models))
    # print('-' * 100)
    # model = input('Please enter the model you want to run:')
    import time
    
    args = parse_args()

    s = time.time()
    if model in kg_embedding_models:
        conf = ModelConf('./conf/kg_embedding/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)
    rec = KnowledgeGraphEmbedding(args, conf)
    rec.train_and_eval()
    e = time.time()
    print("Running time: %f s" % (e - s))

