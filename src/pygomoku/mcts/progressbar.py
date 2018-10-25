import sys
import time


class ProgressBar(object):
    def __init__(self, total_iter, total_sharp=50):
        self.__total_iter = total_iter
        self.__total_sharp = total_sharp
        self.__curr_iter = 0
        self.__time_per_iter = None
        self.__iter_start_time = None

    def __len__(self):
        return self.__total_iter

    def iterStart(self):
        self.__iter_start_time = time.time()
        if self.__curr_iter == 0:
            self.__printProgress()

    def iterEnd(self, additional_msg=''):
        time_interval = time.time() - self.__iter_start_time
        self.__printProgress(additional_msg)
        self.__curr_iter += 1

        if self.__time_per_iter is None:
            self.__time_per_iter = float(time_interval)
        else:
            self.__time_per_iter = (
                (self.__curr_iter-1) * self.__time_per_iter + time_interval) / self.__curr_iter
        if self.__curr_iter == self.__total_iter:
            sys.stdout.write('\n')
            sys.stdout.flush()
            print("Finish!")

    def __printProgress(self, additional_msg=''):
        curr_progress = (self.__curr_iter+1) * 1.0 / self.__total_iter
        curr_sharp_num = int(curr_progress * self.__total_sharp)

        if self.__time_per_iter is not None:
            remain_sec = int(self.__time_per_iter * \
                (self.__total_iter - self.__curr_iter))
            second = remain_sec % 60
            hour = remain_sec // 3600
            minute = (remain_sec % 3600) // 60

        sys.stdout.write(' ' * (self.__total_sharp+100) + '\r')
        sys.stdout.flush()
        if self.__time_per_iter is not None:
            sys.stdout.write('(%.2f%%)[' % (curr_progress*100) + '#' * curr_sharp_num + ' ' * (
                self.__total_sharp - curr_sharp_num) + '] remaining time: %d h %d m %d s\t' % (hour, minute, second) + additional_msg + '\r')
        else:
            sys.stdout.write('(%.2f%%)[' % (curr_progress*100) + '#' * curr_sharp_num + ' ' * (
                self.__total_sharp - curr_sharp_num) + '] remaining time: N/A h N/A m N/A s\t' + additional_msg + '\r')
        sys.stdout.flush()
