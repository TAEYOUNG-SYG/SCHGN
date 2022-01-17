import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='Run Recipe Recommendation.')

    # ------------------------- experimental settings specific for data set --------------------------------------------
    parser.add_argument(
        "--data_path", nargs="?", default="/data/mmc_syg/Allrecipe_processed/", help="Input data path."
    )
    parser.add_argument(
        "--dataset", nargs="?", default="data", help="Choose a dataset."
    )
    parser.add_argument("--emb_size", type=int, default=64, help="Embedding size.")
    parser.add_argument(
        "--regs",
        nargs="?",
        default="1e-5",
        help="Regularization for user and item embeddings.",
    )
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=2, help="gpu id")

    # ------------------------- experimental settings specific for recommender -----------------------------------------
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate recommender."
    )

    # ------------------------- experimental settings specific for sampler ---------------------------------------------
    parser.add_argument(
        "--in_channel", type=str, default="[64, 32]", help="input channels for gcn"
    )
    parser.add_argument(
        "--out_channel", type=str, default="[32, 64]", help="output channels for gcn"
    )

    # ------------------------- experimental settings specific for recommender --------------------------------------------
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="batch size for training."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=1024, help="batch size for test"
    )
    parser.add_argument("--num_threads", type=int, default=4, help="number of threads.")
    parser.add_argument("--epoch", type=int, default=20, help="Number of epoch.")
    parser.add_argument("--show_step", type=int, default=3, help="test step.")
    parser.add_argument("--val_verbose", type=int, default=5, help="test step.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/best_fm.ckpt",
        help="path for pretrain model",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./weights/", help="output directory for model"
    )
    # ------------------------- experimental settings specific for testing ---------------------------------------------
    parser.add_argument(
        "--Ks", nargs="?", default="[20, 40, 60, 80, 100]", help="evaluate K list"
    )
    parser.add_argument(
        "--lr_update", type=int, default=10, help='every xx epochs to decay learning rate'
    )
    parser.add_argument(
        "--gamma", type=float, default=0.1, help='every xx epochs to decay learning rate'
    )
    parser.add_argument(
        "--model", type=str, default='ours', help='choose the model ours/mf/fm'
    )
    parser.add_argument(
        "--reg_image", type=float, default=0.01, help='choose the model ours/mf/fm'
    )
    parser.add_argument(
        "--reg_w", type=float, default=0.1, help='choose the model ours/mf/fm'
    )
    parser.add_argument(
        "--reg_g", type=float, default=0.1, help='choose the model ours/mf/fm'
    )
    parser.add_argument(
        "--reg_health", type=float, default=0.1, help='choose the model ours/mf/fm'
    )
    parser.add_argument(
        "--ssl", type=float, default=0.1, help='choose the model ours/mf/fm'
    )
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")

    return parser.parse_args()
