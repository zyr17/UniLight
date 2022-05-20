import time
import os
import sys
import socket
import shutil

from mainfunc import *
from utils.arguments import parse_args
from utils.log import log, loginit, logexit


def main_wrapper(args):
    args = vars(parse_args(args))
    args = loginit(args)
    log(args, level = 'ALL')
    M = args['main']

    def cleancheck(args):
        clean_flag = False
        if args['clean_logs']:
            clean_flag = True
            if not args['force_clean']:
                log('run over, try to clean log folder `%s`. are you sure to '
                    'clean? if not, press Ctrl+C' % args['log_folder'])
                try:
                    input()
                except KeyboardInterrupt:
                    clean_flag = False
                except EOFError:
                    log('got EOF, stop cleaning', level = 'WARN')
                    clean_flag = False
        lf = os.path.split(args['log_folder'])
        if len(lf[1]) == 0:  # last folder name not split
            lf = os.path.split(lf[0])
        assert len(lf[1]) != 0
        lf = lf[1]
        os.makedirs('./results/cleaned', exist_ok = True)
        srcfile = '%s/main.log' % args['log_folder']
        destfile = './results/%s%s_%s.log' % (
            'cleaned/' if clean_flag else '',
            lf,
            socket.gethostname()
        )
        open(destfile, 'w').write(open(srcfile).read())
        if clean_flag:
            logexit()
            if 'linux' in sys.platform:
                os.system('rm -r %s' % args['log_folder'])
            else:
                print('[ERROR] not in linux, cannot remove log folder! please '
                      'remove it manually.')

    try:
        if M.lower() == 'dqn':
            main = DQNMain(**args)
        elif M.lower() == 'determined':
            main = DeterminedMain(**args)
        elif M in globals():
            main = globals()[M](**args)
        else:
            M = M + 'Main'
            if M in globals():
                main = globals()[M](**args)
            else:
                raise NotImplementedError('unknown main ' + M[:-4])
        main.main()
    except (Exception, KeyboardInterrupt) as e:
        try:
            del main
        except Exception:
            pass
        time.sleep(0.3)
        log('some error cooured! will show below.', level = 'ERROR')
        cleancheck(args)
        raise e
    del main
    time.sleep(0.3)
    cleancheck(args)


if __name__ == "__main__":
    main_wrapper(sys.argv)
