import os
import sys
import time
import pdb
import socket


def gettime():
    dt = ['%02d' % x for x in time.localtime()]
    return '_'.join(dt[:6] + ['%06d' % (int(time.time() * 1000000) % 1000000)])


def get_call_func(frame):
    if frame.f_code.co_filename == sys._getframe().f_code.co_filename:
        return get_call_func(frame.f_back)
    return (frame.f_code.co_filename + ':'
            + frame.f_code.co_name + ':'
            + str(frame.f_code.co_firstlineno))


class Logger:
    def __init__(self):
        self.print_level = ['ERROR', 'WARN', 'INFO']
        self.log_folder = ''

    def init(self, kwargs):
        args = kwargs.copy()
        self.log_folder = (kwargs['work_folder'] + '/'
                           + kwargs['log_folder'] + '/')
        if kwargs['work_folder'] == '' or kwargs['log_folder'] == '':
            self.log_folder = ''
            return kwargs

        subfolder = '_'.join([gettime(), 
                              # kwargs['env'], 
                              kwargs['multi_agent'], 
                              kwargs['agent'], 
                              kwargs['model'], 
                              kwargs['tensorboardx_comment']])
        self.log_folder += subfolder + '/'
        kwargs['log_folder'] = self.log_folder

        os.makedirs(self.log_folder)
        self.main_log = open(self.log_folder + 'main.log', 'w')

        gitdata = 'unknown'
        try:
            gitdata = os.popen(
                'git log --pretty=oneline -n 1').readlines()[0].strip()
        except Exception:
            pass

        self.log('host name:', socket.gethostname(), level = 'TRACE')
        self.log('log folder:', self.log_folder, level = 'INFO')
        self.log('GIT STATUS: ' + gitdata, level = 'INFO')
        self.log('command:', sys.argv, level = 'INFO')

        # save used files
        inf = self.log_folder + 'input_files/'
        os.makedirs(inf)
        if len(kwargs['config']) > 0:
            conf = inf + 'configs/'
            os.mkdir(conf)
            for config in kwargs['config']:
                with open(conf + os.path.split(config)[1], 'w') as f:
                    f.write(open(config).read())

        if kwargs['cityflow_config'] != '':
            cf_config = open(kwargs['cityflow_config']['CONFIG_FILE']).read()
            open(inf + 'cityflow-config.yml', 'w').write(cf_config)
            cf_config = kwargs['cityflow_config']
            with open(inf + 'flow.json', 'w') as f:
                f.write(open(kwargs['work_folder'] 
                             + cf_config['FLOW_FILE']).read())
            with open(inf + 'roadnet.json', 'w') as f:
                f.write(open(kwargs['work_folder'] 
                             + cf_config['ROADNET_FILE']).read())

        return kwargs

    def log(self, *msg, level):
        msg = ' '.join([str(x) for x in msg])
        logstr = '[' + level
        while len(logstr) < 6:
            logstr += ' '
        logstr += ']'
        if level in self.print_level:
            print(logstr, msg)
        logstr += time.asctime(time.localtime()) 
        logstr += ' | ' + get_call_func(sys._getframe().f_back)
        if self.log_folder != '':
            self.main_log.write(logstr + '\n')
            self.main_log.write('       ' + msg + '\n')
            self.main_log.flush()

    def logData(self, data, filename):
        if self.log_folder == '':
            self.log('log folder unset!', level = 'ERROR')
        self.log('save file `%s`' % filename, level = 'FILE')
        raise NotImplementedError()


def loginit(kwargs):
    if '_logger' not in globals():
        globals()['_logger'] = None
    global _logger
    if not _logger:
        _logger = Logger()
    return _logger.init(kwargs)


def log(*msg, level = 'INFO'):
    _logger.log(*msg, level = level)


def logdata(data, filename):
    _logger.logData(data, filename)


def logexit():
    if '_logger' in globals():
        global _logger
        del _logger
