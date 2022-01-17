import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.utils.data as data
import random
import pickle


def get_neg_ingre(ingre_set, ingre_size):
    ingre_id = random.randint(0, ingre_size-1)
    while ingre_id in ingre_set:
        ingre_id = random.randint(0, ingre_size-1)
    return ingre_id


class TrainGenerator(data.Dataset):
    def __init__(self, samples, args_config, dataset):
        super(TrainGenerator, self).__init__()
        self.args_config = args_config
        self.dataset = dataset
        self.n_ingredients = 33147
        self.max_len = 20
        self.masked_p = 0.2
        self._user_input, self._item_input_pos, self._ingre_input_pos, self._ingre_num_pos, self._image_input_pos = samples

    def __len__(self):
        return len(self._user_input)

    def __getitem__(self, index):
        # users,
        # pos_items, pos_image, pos_hl, pos_cate,
        # neg_items, neg_image, neg_hl, neg_cate,

        out_dict = {}
        u_id = self._user_input[index]
        out_dict['u_id'] = u_id
        pos_i_id = self._item_input_pos[index]
        out_dict['pos_i_id'] = pos_i_id
        out_dict['pos_ingre_code'] = self._ingre_input_pos[index]
        out_dict['pos_ingre_num'] = self._ingre_num_pos[index]
        out_dict['pos_img'] = self._image_input_pos[index]
        out_dict['pos_hl'] = self.dataset.item_hl[pos_i_id]
        # out_dict['pos_ingre_emb'] = self.dataset.ingre_emb[pos_i_id]

        out_dict['masked_ingre_seq'], out_dict['pos_ingre_seq'], out_dict['neg_ingre_seq'] = self.ssl_task(out_dict['pos_ingre_code'], out_dict['pos_ingre_num'])

        pos_items = self.dataset.trainList[u_id]
        pos_validTest = self.dataset.validTestRatings[u_id]
        neg_i_id = self.get_random_neg(pos_items, pos_validTest)
        out_dict['neg_i_id'] = neg_i_id
        out_dict['neg_ingre_code'] = self.dataset.ingredientCodeDict[neg_i_id]
        out_dict['neg_ingre_num'] = self.dataset.ingredientNum[neg_i_id]
        out_dict['neg_img'] = self.dataset.embImage[neg_i_id]
        out_dict['neg_hl'] = self.dataset.item_hl[neg_i_id]
        # out_dict['neg_ingre_emb'] = self.dataset.ingre_emb[neg_i_id]

        return out_dict

    def ssl_task(self, ingre_seq, ingre_num):
        masked_ingre_seq = []
        neg_ingre = []
        pos_ingre = []
        ingre_set = set(ingre_seq[:ingre_num])
        for idx, ingre in enumerate(ingre_seq):
            if idx < ingre_num:
                pos_ingre.append(ingre)
                prob = random.random()
                if prob < self.masked_p:
                    masked_ingre_seq.append(self.n_ingredients+1)
                    neg_ingre.append(get_neg_ingre(ingre_set, self.n_ingredients))
                else:
                    masked_ingre_seq.append(ingre)
                    neg_ingre.append(ingre)
            else:
                pos_ingre.append(ingre)
                masked_ingre_seq.append(ingre)
                neg_ingre.append(ingre)

        assert len(masked_ingre_seq) == self.max_len
        assert len(pos_ingre) == self.max_len
        assert len(neg_ingre) == self.max_len

        return torch.tensor(masked_ingre_seq, dtype=torch.long), \
               torch.tensor(pos_ingre, dtype=torch.long), \
               torch.tensor(neg_ingre, dtype=torch.long)

    def get_random_neg(self, train_pos, validTest_pos):
        while True:
            neg_i_id = np.random.randint(self.dataset.num_items)
            if neg_i_id not in train_pos and neg_i_id not in validTest_pos:
                break
        return neg_i_id


class CFData(object):
    """
    Loading the interaction data file
    """
    def __init__(self, args_config):
        self.args_config = args_config
        path = args_config.data_path + 'data'
        self.user_range = []
        self.item_range = []
        self.n_users, self.n_items, self.n_train, self.n_valid, self.n_test = 0, 0, 0, 0, 0

        self.trainMatrix = self.load_training_file_as_matrix(path + '.train.rating')
        self.trainList = self.load_training_file_as_list(path + '.train.rating')
        self.testRatings = self.load_training_file_as_list(path + '.test.rating')
        self.testNegatives = self.load_negative_file(path + '.test.negative')
        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape

        self.validRatings, self.valid_users = self.load_valid_file_as_list(path + '.valid.rating')
        self.validNegatives = self.load_negative_file(path + '.valid.negative')
        self.validTestRatings = self.load_valid_test_file_as_dict(path + '.valid.rating', path + '.test.rating')
        self.cold_list, self.cold_num, self.train_item_list = self.get_cold_start_item_num()

        self.train_data = self.generate_interactions(path + '.train.rating')
        self.valid_data = self.generate_interactions(path + '.valid.rating')
        self.test_data = self.generate_interactions(path + '.test.rating')

        self.train_user_dict, self.valid_user_dict, self.test_user_dict = self.generate_user_dict()
        self.statistic_interactions()

    def load_valid_file_as_list(self, filename):
        lists, items, user_list = [], [], []
        with open(filename, 'r') as f:
            line = f.readline()
            index = 0
            last_u = int(line.split('\t')[0])
            while line is not None and line != '':
                arr = line.split('\t')
                u, i = int(arr[0]), int(arr[1])
                if last_u < u:
                    index = 0
                    lists.append(items)
                    user_list.append(last_u)
                    items = []
                    last_u = u
                index += 1
                items.append(i)
                line = f.readline()
        lists.append(items)
        user_list.append(u)
        return lists, user_list

    def load_valid_test_file_as_dict(self, valid_file, test_file):

        validTestRatings = {}
        for u in range(self.num_users):
            validTestRatings[u] = set()

        fv = open(valid_file, 'r')
        for line in fv:
            arr = line.split('\t')
            u, i = int(arr[0]), int(arr[1])
            validTestRatings[u].add(i)
        fv.close()

        ft = open(test_file, 'r')
        for line in ft:
            arr = line.split('\t')
            u, i = int(arr[0]), int(arr[1])
            validTestRatings[u].add(i)
        ft.close()

        return validTestRatings

    def load_training_file_as_list(self, filename):
        u_ = 0
        lists, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    lists.append(items)
                    items = []
                    u_ += 1
                index += 1
                items.append(i)
                line = f.readline()
        lists.append(items)
        return lists

    def load_training_file_as_matrix(self, filename):
        num_users, num_items = 0, 0
        with open(filename, 'r') as f:
            line = f.readline()
            while line is not None and line != '':
                arr = line.split('\t')
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, 'r') as f:
            line = f.readline()
            while line is not None and line != '':
                arr = line.split('\t')
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    @staticmethod
    def generate_interactions(filename):
        inter_mat = []
        lines = open(filename, 'r').readlines()
        for line in lines:
            tokens = line.strip().split('\t')
            u_id, pos_id = int(tokens[0]), int(tokens[1])
            inter_mat.append([u_id, pos_id])
        return np.array(inter_mat)

    def generate_user_dict(self):
        def generate_dict(inter_mat):
            user_dict = defaultdict(list)
            for u_id, i_id in inter_mat:
                user_dict[u_id].append(i_id)
            return user_dict

        num_users = max(max(self.train_data[:, 0]), max(self.valid_data[:, 0]), max(self.test_data[:, 0])) + 1

        self.train_data[:, 1] = self.train_data[:, 1] + num_users
        self.valid_data[:, 1] = self.valid_data[:, 1] + num_users
        self.test_data[:, 1] = self.test_data[:, 1] + num_users

        train_user_dict = generate_dict(self.train_data)
        valid_uesr_dict = generate_dict(self.valid_data)
        test_user_dict = generate_dict(self.test_data)

        return train_user_dict, valid_uesr_dict, test_user_dict

    def statistic_interactions(self):
        def id_range(train_mat, valid_mat, test_mat, idx):
            min_id = min(min(train_mat[:, idx]), min(valid_mat[:, idx]), min(test_mat[:, idx]))
            max_id = max(max(train_mat[:, idx]), max(valid_mat[:, idx]), max(test_mat[:, idx]))
            n_id = max_id - min_id + 1
            return (min_id, max_id), n_id

        self.user_range, self.n_users = id_range(
            self.train_data, self.valid_data, self.test_data, idx=0
        )
        self.item_range, self.n_items = id_range(
            self.train_data, self.valid_data, self.test_data, idx=1
        )
        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_data)
        self.n_test = len(self.test_data)

        print("-" * 50)
        print("-     user_range: (%d, %d)" % (self.user_range[0], self.user_range[1]))
        print("-     item_range: (%d, %d)" % (self.item_range[0], self.item_range[1]))
        print("-        n_train: %d" % self.n_train)
        print("-        n_valid: %d" % self.n_valid)
        print("-         n_test: %d" % self.n_test)
        print("-        n_users: %d" % self.n_users)
        print("-        n_items: %d" % self.n_items)
        print("-        n_cold: %d" % self.cold_num)
        print("-" * 50)

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, 'r') as f:
            line = f.readline()
            while line is not None and line != '':
                arr = line.split('\t')
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x) + self.n_users)
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def get_cold_start_item_num(self):
        train_item_list = []
        for i_list in self.trainList:
            train_item_list.extend(i_list)
        test_item_list = []
        for r in self.testRatings:
            test_item_list.extend(r)
        valid_item_list = []
        for r in self.validRatings:
            valid_item_list.extend(r)
        c_list = list((set(test_item_list) | set(valid_item_list)) - set(train_item_list))
        t_list = list(set(train_item_list))
        return c_list, len(c_list), len(t_list)


class AGData(object):
    def __init__(self, args_config, item_start_id, entity_start_id, relation_start_id):
        self.args_config = args_config
        self.item_start_id = item_start_id
        self.entity_start_id = entity_start_id
        self.relation_start_id = relation_start_id

        self.path = args_config.data_path
        ag_file = self.path + 'ag_final.txt'

        self.entity_range, self.n_entities = [], 0
        self.node_range, self.n_nodes = [], 0
        self.relation_range, self.n_relations = [], 0
        self.n_ag_triples = 0
        self.ingre_range = []

        self.embImage = np.load(self.path + 'data_image_features_float.npy')
        self.image_size = self.embImage.shape[1]

        self.num_ingredients = 33147
        self.ingre_embed = np.load('/home/mmc_syg/projects/recipe_recommendation/ssl/ingre_emb.npy')
        self.ingredientNum = self.load_id_ingredient_num(self.path + 'data_id_ingre_num_file')
        self.ingredientCodeDict = np.load(self.path + 'data_ingre_code_file.npy')
        self.ingredientCode = self.load_ingre_code(self.path + 'data_ingre_code_file.npy')

        self.ag_data = self.load_ag(ag_file)
        self.item_hl, self.item_cate = self.load_dict()
        self.statistic_ag_triples()

    def load_dict(self):
        with open('/home/mmc_syg/projects/recipe_recommendation/item_cate_dict.pkl', 'rb') as f:
            item_cate = pickle.load(f)
        with open('/home/mmc_syg/projects/recipe_recommendation/new_item_hl_dict.pkl', 'rb') as f:
            item_hl = pickle.load(f)
        return item_hl, item_cate

    def load_ingre_code(self, filename):
        ingre_code = np.load(filename) + self.entity_start_id
        return ingre_code

    def load_id_ingredient_num(self, filename):
        fr = open(filename, 'r')
        ingredientNumList = []
        for line in fr:
            arr = line.strip().split('\t')
            ingredientNumList.append(int(arr[1]))
        return ingredientNumList

    def load_ag(self, filename):
        def remap_ag_id(ori_ag_np, first_start, sec_start, third_start):
            new_ag_np = ori_ag_np.copy()
            # [0, #item] --> [#user, #user + #item]
            new_ag_np[:, 0] = ori_ag_np[:, 0] + first_start
            # [0, #entity] --> [#user+#item, #user+#item+entity]
            new_ag_np[:, 2] = ori_ag_np[:, 2] + third_start
            # [0, 3] --> [2, 5]
            new_ag_np[:, 1] = ori_ag_np[:, 1] + sec_start
            return new_ag_np

        can_ag_np = np.loadtxt(filename, dtype=np.int)
        can_ag_np = can_ag_np[can_ag_np[:, 1] == 0]
        can_ag_np = remap_ag_id(can_ag_np, self.item_start_id, self.relation_start_id, self.entity_start_id)
        health_ag = np.loadtxt(self.path + 'new_hl.txt', dtype=np.int)
        health_np = remap_ag_id(health_ag, self.item_start_id, self.relation_start_id, self.entity_start_id+self.num_ingredients)
        ag_np = np.concatenate((can_ag_np, health_np), axis=0)

        return ag_np

    def statistic_ag_triples(self):
        def id_range(ag_mat, idx):
            min_id = min(min(ag_mat[:, idx]), min(ag_mat[:, 2 - idx]))
            max_id = max(max(ag_mat[:, idx]), max(ag_mat[:, 2 - idx]))
            n_id = max_id - min_id + 1
            return (min_id, max_id), n_id

        self.node_range, self.n_nodes = id_range(self.ag_data, idx=0)
        self.node_range = list(self.node_range)
        self.node_range[0] = 0
        self.n_nodes = self.node_range[1] + 1
        self.relation_range, self.n_relations = id_range(self.ag_data, idx=1)
        self.n_ag_triples = len(self.ag_data)
        print("-" * 50)
        print(
            "-   node_range: (%d, %d)" % (self.node_range[0], self.node_range[1])
        )
        print(
            "-   relation_range: (%d, %d)"
            % (self.relation_range[0], self.relation_range[1])
        )
        print("-   n_nodes: %d" % self.n_nodes)
        print("-   n_relations: %d" % self.n_relations)
        print("-   n_kg_triples: %d" % self.n_ag_triples)
        print("-" * 50)


class CAGData(CFData, AGData):
    def __init__(self, args_config):
        CFData.__init__(self, args_config=args_config)
        AGData.__init__(
            self,
            args_config=args_config,
            item_start_id=self.n_users,
            entity_start_id=self.n_users + self.n_items,
            relation_start_id=2
        )
        self.args_config = args_config
        self.g2i_index, self.i2u_index = self.combine_cf_ag()

    def combine_cf_ag(self):
        ag_mat = self.ag_data
        cf_mat = self.train_data
        i2u_index = []
        g2i_index = []
        print('Begin to load interaction triples ... ')
        for uid, iid in tqdm(cf_mat, ascii=True):
            i2u_index.append([iid, uid])

        print('\nBegin to load attribute graph triples ... ')
        for h_id, r_id, t_id in tqdm(ag_mat, ascii=True):
            g2i_index.append([t_id, h_id])

        print('num of i2u_index: {}'.format(len(i2u_index)))
        print('num of g2i_index: {}'.format(len(g2i_index)))
        return g2i_index, i2u_index


def build_loader(samples, args_config, dataset):
    train_generator = TrainGenerator(samples=samples, args_config=args_config, dataset=dataset)
    train_loader = data.DataLoader(
        train_generator,
        batch_size=args_config.batch_size,
        shuffle=True,
        num_workers=args_config.num_threads
    )

    return train_loader


if __name__ == '__main__':
    pass

