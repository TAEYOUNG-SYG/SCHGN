import os
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import random
import math

import torch
import numpy as np
import shutil

import logging
from time import time
from copy import deepcopy
from tqdm import tqdm
from new_model import MyModel

from torch.utils.tensorboard import SummaryWriter
from config import arg_parser
from new_dataset import CAGData, build_loader
from utils import AverageMeter, TimeMeter
import datetime
import pickle


def sampling(dataset):
    _user_input, _item_input_pos, _ingre_input_pos, _ingre_num_pos, _image_input_pos = [], [], [], [], []
    for (u, i) in dataset.trainMatrix.keys():
        _user_input.append(u)
        _item_input_pos.append(i)
        _ingre_input_pos.append(dataset.ingredientCodeDict[i])
        _ingre_num_pos.append(dataset.ingredientNum[i])
        _image_input_pos.append(dataset.embImage[i])
    return _user_input, _item_input_pos, _ingre_input_pos, _ingre_num_pos,  _image_input_pos


def train_one_epoch(
        model,
        train_loader,
        model_optim,
        g2i_edge_matrix,
        i2u_edge_matrix,
        cur_epoch,
        device,
        logger
):
    loss_meter = AverageMeter()
    base_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    ssl_loss_meter = AverageMeter()
    time_meter = TimeMeter()

    num_batch = len(train_loader)
    for ind, batch_data in enumerate(train_loader):
        if torch.cuda.is_available():
            batch_data = {k: v.to(device, non_blocking=True) for k, v in batch_data.items()}

        model_optim.zero_grad()

        neg_id = batch_data['neg_i_id']
        neg_img = batch_data['neg_img'].float()
        neg_ingre_num = batch_data['neg_ingre_num']
        neg_ingre_code = batch_data['neg_ingre_code']
        neg_hl = batch_data['neg_hl'].long()


        pos_id = batch_data['pos_i_id']
        pos_img = batch_data['pos_img'].float()
        pos_ingre_num = batch_data['pos_ingre_num']
        pos_ingre_code = batch_data['pos_ingre_code']
        pos_hl = batch_data['pos_hl'].long()

        masked_ingre_seq = batch_data['masked_ingre_seq']
        pos_ingre_seq = batch_data['pos_ingre_seq']
        neg_ingre_seq = batch_data['neg_ingre_seq']

        users = batch_data['u_id']

        loss_batch, base_loss_batch, reg_loss_batch, ssl_loss_batch = model(
            users,
            pos_id, pos_ingre_code, pos_ingre_num, pos_img, pos_hl,
            neg_id, neg_ingre_code, neg_ingre_num, neg_img, neg_hl,
            g2i_edge_matrix, i2u_edge_matrix,
            masked_ingre_seq, pos_ingre_seq, neg_ingre_seq
        )

        loss_batch.backward()
        model_optim.step()

        loss_meter.update(loss_batch.item())
        base_loss_meter.update(base_loss_batch.item())
        reg_loss_meter.update(reg_loss_batch.item())
        ssl_loss_meter.update(ssl_loss_batch.item())
        time_meter.update()

        if ind % 100 == 0:
            logger.info('Epoch %d, Batch %d / %d, loss = %.4f, base_loss = %.4f,  ssl_loss = %.4f, %.3f seconds/batch' % (
                cur_epoch, ind, num_batch, loss_meter.avg, base_loss_meter.avg, ssl_loss_meter.avg, 1.0 / time_meter.avg
            ))
    return loss_meter.avg, base_loss_meter.avg


def train(train_loader, graph, data_config, args_config, device, writer, logger):
    logger.info('------- {} --------'.format(args_config.model))
    model = MyModel(data_config=data_config, args_config=args_config, device=device)
    model = model.to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=args_config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=args_config.lr_update, gamma=args_config.gamma)

    max_valid_auc = 0
    best_valid_res = {}
    best_model_state = None
    g2i_edge_matrix = torch.tensor(graph.g2i_index).to(device, dtype=torch.long)
    i2u_edge_matrix = torch.tensor(graph.i2u_index).to(device, dtype=torch.long)

    for epoch in range(args_config.epoch):
        model.train()
        cur_epoch = epoch + 1
        loss_e, base_loss_e = train_one_epoch(
            model,
            train_loader,
            model_optimizer,
            g2i_edge_matrix,
            i2u_edge_matrix,
            cur_epoch,
            device,
            logger
        )
        logger.info('Epoch %d, loss = %.4f, base_loss = %.4f' % (cur_epoch, loss_e, base_loss_e))

        scheduler.step()
        logger.info('current lr: {:.5f}'.format(scheduler.get_last_lr()[0]))

        if cur_epoch % args_config.val_verbose == 0:
            model.eval()
            with torch.no_grad():
                start = time()
                valid_feed_dicts = init_eval(graph, phase='val')
                stop_init = time()
                recall, ndcg, auc = evaluate(model, graph, valid_feed_dicts, 'val', device, g2i_edge_matrix, i2u_edge_matrix, args_config, logger)
                stop_eval = time()
                logger.info('init time: {:.4f}'.format(stop_init-start))
                logger.info('evalate time: {:.4f}'.format(stop_eval-stop_init))
                for idx, (recall_k, ndcg_k, auc_k) in enumerate(zip(recall, ndcg, auc)):
                    res = 'k = {}: Recall = {:.4f}, NDCG = {:.4f}, AUC = {:.4f}'.format((idx + 1) * 10, recall_k, ndcg_k, auc_k)
                    logger.info(res)
                if max_valid_auc < auc[-1]:
                    max_valid_auc = auc[-1]
                    best_valid_res['res'] = (recall, ndcg, auc)
                    best_valid_res['epoch'] = cur_epoch
                    best_model_state = deepcopy(model.state_dict())

    logger.info('Epoch {} is the best epoch for validation set'.format(best_valid_res['epoch']))
    for idx, (recall_k, ndcg_k, auc_k) in enumerate(zip(*best_valid_res['res'])):
        res = 'k = {}: Recall = {:.4f}, NDCG = {:.4f}, AUC = {:.4f}'.format((idx + 1) * 10, recall_k, ndcg_k, auc_k)
        logger.info(res)
    logger.info('Start Testing...')

    with torch.no_grad():
        test_feed_dicts = init_eval(graph, phase='test')
        model.load_state_dict(best_model_state)
        model.eval()
        recall, ndcg, auc = evaluate(model, graph, test_feed_dicts, 'test', device, g2i_edge_matrix, i2u_edge_matrix, args_config, logger)
        logger.info('Testing Done...')
        for idx, (recall_k, ndcg_k, auc_k) in enumerate(zip(recall, ndcg, auc)):
            res = 'k = {}: Recall = {:.4f}, NDCG = {:.4f}, AUC = {:.4f}'.format((idx+1)*10, recall_k, ndcg_k, auc_k)
            logger.info(res)


def init_eval(dataset, phase='val'):
    if phase == 'val':
        valid_users = dataset.valid_users
        for idx in range(len(valid_users)):
            user = valid_users[idx]
            pos_item_list = dataset.validRatings[idx]
            neg_item_list = dataset.validNegatives[idx]
            for pos_item in pos_item_list:
                if pos_item in neg_item_list:
                    neg_item_list.remove(pos_item)
            item_input = neg_item_list + pos_item_list
            user_input = np.full(len(item_input), user, dtype='int')
            item_input = np.array(item_input)
            img_input = []
            ingre_input = []
            ingre_num_input = []
            hl_input = []
            for item in item_input:
                img_input.append(dataset.embImage[item])
                ingre_num_input.append(dataset.ingredientNum[item])
                ingre_input.append(dataset.ingredientCodeDict[item])
                hl_input.append(dataset.item_hl[item])
            img_input = np.array(img_input)
            ingre_num_input = np.array(ingre_num_input)
            ingre_input = np.array(ingre_input)
            hl_input = np.array(hl_input)
            user_batch = {
                'user_input': torch.tensor(user_input).long(), 'item_input': torch.tensor(item_input).long(),
                'img_input': torch.tensor(img_input).float(), 'ingre_num_input': torch.tensor(ingre_num_input).long(),
                'ingre_input': torch.tensor(ingre_input).long(), 'hl_input': torch.tensor(hl_input).long(),
            }
            yield user_batch
    else:
        for user in range(dataset.num_users):
            pos_item_list = dataset.testRatings[user]
            neg_item_list = dataset.testNegatives[user]
            for pos_item in pos_item_list:
                if pos_item in neg_item_list:
                    neg_item_list.remove(pos_item)
            item_input = neg_item_list + pos_item_list
            user_input = np.full(len(item_input), user, dtype='int')
            item_input = np.array(item_input)
            img_input = []
            ingre_input = []
            ingre_num_input = []
            hl_input = []
            for item in item_input:
                img_input.append(dataset.embImage[item])
                ingre_num_input.append(dataset.ingredientNum[item])
                ingre_input.append(dataset.ingredientCodeDict[item])
                hl_input.append(dataset.item_hl[item])
            img_input = np.array(img_input)
            ingre_num_input = np.array(ingre_num_input)
            ingre_input = np.array(ingre_input)
            hl_input = np.array(hl_input)
            user_batch = {
                'user_input': torch.tensor(user_input).long(), 'item_input': torch.tensor(item_input).long(),
                'img_input': torch.tensor(img_input).float(), 'ingre_num_input': torch.tensor(ingre_num_input).long(),
                'ingre_input': torch.tensor(ingre_input).long(), 'hl_input':torch.tensor(hl_input).long(),
            }
            yield user_batch


def evaluate(model, dataset, feed_dicts, phase, device, g2i_edge_matrix, i2u_edge_matrix, args_config, logger):
    res = []
    user_idx = 0
    for user_batch in tqdm(feed_dicts, ascii=True, desc='evaluating'):
        if phase == 'val':
            pos_items = dataset.validRatings[user_idx]
        else:
            pos_items = dataset.testRatings[user_idx]
        r, n, a = eval_by_user(model, pos_items, user_batch, device, g2i_edge_matrix, i2u_edge_matrix, args_config)
        res.append((r, n, a))
        user_idx += 1
    res = np.array(res)
    recall, ndcg, auc = (res.mean(axis=0)).tolist()
    return recall, ndcg, auc


def eval_by_user(model, gt_items, user_batch, device, g2i_edge_matrix, i2u_edge_matrix, args_config):
    user_batch = {k: v.to(device, non_blocking=True) for k, v in user_batch.items()}

    user_input = user_batch['user_input']
    item_input = user_batch['item_input']
    img_input = user_batch['img_input']
    ingre_num_input = user_batch['ingre_num_input']
    ingre_input = user_batch['ingre_input']
    hl_input = user_batch['hl_input']

    predictions, u_final_emb = model.inference(user_input, item_input, ingre_input, ingre_num_input, img_input, hl_input, g2i_edge_matrix, i2u_edge_matrix)

    predictions = predictions.cpu().numpy()

    pos_num = len(gt_items)
    # neg_pred, pos_pred = predictions[:-pos_num], predictions[-pos_num:]
    gt_idx = range(len(predictions)-pos_num, len(predictions), 1)

    pred_idx = np.argsort(predictions)[::-1]
    recall, ndcg, auc = [], [], []
    neg_num = 500
    try:
        auc_value = get_auc_fast(gt_idx, predictions, neg_num)
    except ZeroDivisionError:
        print(gt_items)
        print(user_input)
        print(predictions.shape)

    for k in range(10, 51, 10):
        selected_idx = pred_idx[:k]
        rec_val, ndcg_val = metrics(selected_idx, gt_idx)
        recall.append(rec_val)
        ndcg.append(ndcg_val)
        auc.append(auc_value)
    return recall, ndcg, auc


def metrics(doc_list, rel_list):
    dcg = 0.0
    hit_num = 0.0

    for i in range(len(doc_list)):
        if doc_list[i] in rel_list:
            dcg += 1/(math.log(i+2) / math.log(2))
            hit_num += 1

    idcg = 0.0
    for i in range(min(len(doc_list), len(rel_list))):
        idcg += 1/(math.log(i+2) / math.log(2))
    ndcg = dcg/ idcg
    recall = hit_num / len(rel_list)
    return recall, ndcg


def get_auc(rel_list, predictions, neg_num):
    auc_value = 0.0
    for rel in rel_list:
        for pre in predictions[0: neg_num]:
            if predictions[rel] > pre:
                auc_value += 1
    return auc_value / (len(rel_list) * neg_num)


def get_auc_fast(rel_list, predictions, neg_num):
    neg_predictions = predictions[0:neg_num]
    auc_value = np.sum([np.sum(neg_predictions < predictions[idx]) for idx in rel_list])
    return auc_value / (len(rel_list) * neg_num)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + 'model-{}.pt'.format(state['epoch']))
    if is_best:
        shutil.copyfile(prefix + 'model-{}.pt'.format(state['epoch']), prefix + 'model_best.pth.tar')
    logging.info('model saved to %s' % prefix + 'model-{}.pt'.format(state['epoch']))


def main():
    args_config = arg_parser()
    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join('./logs', time_str)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # set the random seed
    SEED = 2020
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = '%(asctime)s %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler()  # 输出到控制台的handler
    chlr.setFormatter(formatter)

    fhlr = logging.FileHandler(log_dir+'.log')  # 输出到文件的handler
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    # logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    logger.info("Race Start...")
    logger.info(args_config)
    logger.info('learning rate: {} | regs: {}'.format(args_config.lr, args_config.regs))
    device = torch.device('cuda:{}'.format(args_config.gpu_id)) if args_config.use_gpu else torch.device('cpu')

    FData = CAGData(args_config)
    logger.info("obtain the processed dataset")

    data_config = {
        'n_users': FData.n_users,
        'n_items': FData.n_items,
        'item_range': FData.item_range,
        'image_size': FData.image_size,
        'n_cold': FData.cold_num,
        'n_ingredients': FData.num_ingredients,
        'n_nodes': FData.n_nodes
    }

    samples = sampling(FData)
    train_loader = build_loader(samples, args_config, FData)

    train(
        train_loader=train_loader,
        graph=FData,
        data_config=data_config,
        args_config=args_config,
        device=device,
        writer=writer,
        logger=logger
    )

    writer.close()
    logging.info('Model: {}'.format(args_config.model))
    logging.info(
        'Learning rate: {} | reg: {} | reg_image: {} | reg_w: {} | reg_g: {} | ssl: {}'.format(args_config.lr, args_config.regs, args_config.reg_image, args_config.reg_w, args_config.reg_g, args_config.ssl))


if __name__ == '__main__':
    main()
