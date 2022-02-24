import time
import logging
import os
def create_logger(stage):
    log_dir = './result/log/'
    if not os.path.exists(log_dir):
        print('=> creating {}'.format(log_dir))
        os.makedirs(log_dir)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(stage,'log', time_str)
    final_log_file = os.path.join(log_dir,log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger


