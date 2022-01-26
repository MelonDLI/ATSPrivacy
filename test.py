import argparse

def main():

    parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
    # parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
    parser.add_argument('--aug_list', default=None, required=True, type=str, help='Vision model.')
    # parser.add_argument('--rlabel', default=False, type=bool, help='rlabel')
    parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
    parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
    # parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
    opt = parser.parse_args()


    # print(opt.data)
    pathname = 'search/data_{}_arch_{}/{}'.format(opt.data, opt.arch, opt.aug_list)
    print(pathname)



if __name__ == '__main__':
    main()

# import argparse

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

# args = parser.parse_args()
# print(args.accumulate(args.integers))