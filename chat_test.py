import os
from kogpt2_chat import KoGPT2Chat, get_kogpt2_args

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_code():
    args = get_kogpt2_args(is_chat=True)
    model = KoGPT2Chat.load_from_checkpoint(args.model_params)
    while 1:
        q = input('q: ')
        if q == 'quit':
            break
        a = model.chat(q=q)
        print(a)


if __name__ == '__main__':
    test_code()
