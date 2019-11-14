import os
from functools import reduce


def file_extension(path):
  return os.path.splitext(path)[1]


class DirManager:
    def __init__(self, dir_path):
        self.path, self.dir_list, self.file_name_list = list(os.walk(dir_path))[0]  # path，all_dirs，all_filename

    def get_all_file_name(self):
        return reduce(lambda x, y: x + y, [[item[2][i] for i in range(len(item[2]))] for item in
                      os.walk(self.path)])  # return all file abspath (all abs path in this path including subdirs)

    def get_file_abspath(self):
        return [os.path.join(self.path, item) for item in
                self.file_name_list]  # return all file abspath (all abs path in this path without subdirs)

    def __all_file_filter(self, post_fix_set):
        """
        :param post_fix_set:{"jpg", "png", "json", "xml"}
        :return: [[], ['d:file_utils.xml'], ['d:.idea\\encodings.xml', 'd:.idea\\file_manager.png', 'd:.idea\\misc.xml']
        """
        files = self.get_all_file_name()  # todo
        return [list(filter(lambda x: x.split(".")[-1] in post_fix_set, item)) for item in files]

    def all_file_filter(self, post_fix_set):
        """
        :param post_fix_set: post_fix_set:{"jpg", "png", "json", "xml"}
        :return: return_type=2:[['d:file_utils.xml', 'd:.idea\\encodings.xml', 'd:.idea\\file_manager.png',
        'd:.idea\\misc.xml']
        :return return_type=
        """
        file_list = self.__all_file_filter(post_fix_set)
        return reduce(lambda x, y: x + y, file_list)

    def __filt_filter(self, post_fix_set):
        files = self.get_file_abspath()
        return [list(filter(lambda x: x.split(".")[-1] in post_fix_set, item)) for item in files]
        # return list(filter(lambda x: x.split(".")[-1] in post_fix_set, files))

    def file_filter(self, post_fix_set):
        file_list = self.__filt_filter(post_fix_set)
        return reduce(lambda x, y: x + y, file_list)


if __name__ == '__main__':
    a = DirManager("E:\\BaiduNetdiskDownload\\300 Face in Wild dataset")
    a.file_filter({"pts"})