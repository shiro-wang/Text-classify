import os
import matplotlib.pyplot as plt
import logging
from pathlib import Path

def save_result(args, title, result, xlabel="batch", ylabel="loss"):
  
    dir_path = "./results/{args.model_name}" 

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    plt.plot(result)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(['loss'], loc='upper right')
    plt.savefig("{dir_path}/png/{args.max_length}_{args.batch_size}_{args.num_epoch}_{args.learning_rate}_{args.random}.png")
    plt.clf()

def setup_logger(log_path: str):
    l = logging.getLogger('l')


    # create log directory if not exists
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    l.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(
        filename=log_path,
        mode='a'
    )
    streamHandler = logging.StreamHandler()

    allFormatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s"
    )

    fileHandler.setFormatter(allFormatter)
    fileHandler.setLevel(logging.INFO)

    streamHandler.setFormatter(allFormatter)
    streamHandler.setLevel(logging.INFO)

    l.addHandler(streamHandler)
    l.addHandler(fileHandler)

    return l

def save_logger(args, acc):
    dir_path = "./results/{args.model_name}" 

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    log_path = "{dir_path}/results.log"
    logger = setup_logger(args.log_path)
    logger.info("================= Parameters =================")
    logger.info("max_length:{args.max_length}, batch:{args.batch_size}, epoch:{args.num_epoch}, lr:{args.learning_rate}, random{args.random}")
    logger.info("================= Results =================")
    logger.info("Accuracy: {acc}")
    