import argparse


def args_initialization():
    parser = argparse.ArgumentParser(description="Text Classifier")
    
    # 基本参数
    parser.add_argument("-path", type=str,
                        default="data/", help="数据位置")
    parser.add_argument("-lr", type=float, default=2e-5, help="初始学习率 [默认: 2e-5]")
    parser.add_argument("-epochs", type=int, default=1, help="Epoch数 [默认: 1]")
    parser.add_argument("-early-stop", type=int, default=1000,
                        help="早停的Batch数，即经过多少Batch数没有提升则停止训练 [默认: 1000]")
    parser.add_argument("-batch-size", type=int, default=8, help="Batch Size [默认: 8]")
    parser.add_argument("-save-dir", type=str, default="", help="模型存放位置")
    parser.add_argument("-max-length", type=int, default=128, help="最大序列长度 [默认: 128]")
    parser.add_argument("-dropout", type=float, default=0.5, help="dropout率 [默认: 0.5]")
    parser.add_argument("-seed", type=int, default=2021, help="随机种子 [默认: 2021]")
   
    # BERT参数
    parser.add_argument("-bert-pretrained", type=str,
                        default="../model/chinese_roberta", help="BERT预训练模型位置")
    
    
    args = parser.parse_args(args=[])
    return args
