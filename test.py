from argparse import ArgumentParser

description = 'testing for passing multiple arguments and to get list of args'
parser = ArgumentParser(description=description)
# parser.add_argument('--item', action='store', 
#                     type=str, nargs='*', default=['item1', 'item2', 'item3'],
#                     help="Examples: -i item1 item2, -i item3")
parser.add_argument('--defense', action='store',
                    type=str, nargs='*', default=['prune','95'],
                    help="defense type")
opts = parser.parse_args()

print("List of items: {}".format(opts.defense[0]))