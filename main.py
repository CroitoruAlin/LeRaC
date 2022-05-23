from arguments import get_args
from runs import RUNS

if __name__ == '__main__':
    args = get_args()
    RUNS[args.model_name][args.dataset](args)