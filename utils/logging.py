from datetime import datetime


def logger(s):
    return print('{} | {}'.format(datetime.now(), s))
