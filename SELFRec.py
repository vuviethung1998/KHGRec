from data.loader import FileIO

class SELFRec(object):
    def __init__(self, config, args=None):
        self.social_data = []
        self.feature_data = []
        self.config = config
        self.kwargs = {}
        if args:
            self.kwargs = args 
        experiment = self.kwargs['experiment']
        print(experiment)
        if experiment == 'full':
            default_dir = f"./dataset/{config['dataset']}/"
            self.training_data = FileIO.load_data_set(default_dir + config['training.set'], config['model.type'])
            self.test_data = FileIO.load_data_set(default_dir + config['test.set'], config['model.type'])
            self.knowledge_data = FileIO.load_kg_data(default_dir + f"{config['dataset']}.kg")
        elif experiment == 'missing':
            default_dir = f"./dataset/{config['dataset']}/{experiment}/"
            self.training_data = FileIO.load_data_set(default_dir + 'train' + f'_{self.kwargs["missing_pct"]}.txt', config['model.type'])
            self.test_data = FileIO.load_data_set(default_dir + 'test' + f'_{self.kwargs["missing_pct"]}.txt', config['model.type'])
            self.knowledge_data = FileIO.load_kg_data(f"./dataset/{config['dataset']}/" + f"{config['dataset']}.kg")
        elif experiment == 'cold_start':
            default_dir = f"./dataset/{config['dataset']}/{experiment}/"
            self.training_data = FileIO.load_data_set(default_dir + "train.txt", config['model.type'])
            self.test_data = FileIO.load_data_set(default_dir + f"test_group_{self.kwargs['group_id']}.txt", config['model.type'])
            self.knowledge_data = FileIO.load_kg_data(default_dir + f"{config['dataset']}.kg")
        elif experiment == 'add_noise':
            default_dir = f"./dataset/{config['dataset']}/{experiment}/"
            self.training_data = FileIO.load_data_set(default_dir + 'train' + f'_{self.kwargs["noise_pct"]}.txt', config['model.type'])
            self.test_data = FileIO.load_data_set(default_dir + 'test' + f'_{self.kwargs["noise_pct"]}.txt', config['model.type'])
            self.knowledge_data = FileIO.load_kg_data(f"./dataset/{config['dataset']}/" + f"{config['dataset']}.kg")

        print('Reading data and preprocessing...')

    def execute(self):
        # import the model module
        import_str = 'from model.'+ self.config['model.type'] +'.' + self.config['model.name'] + ' import ' + self.config['model.name']
        exec(import_str)
        recommender = self.config['model.name'] + '(self.config,self.training_data,self.test_data,self.knowledge_data,**self.kwargs)'
        eval(recommender).execute()
