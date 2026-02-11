# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import time
import datetime


def get_now_str():
    now_str = datetime.datetime.strftime(datetime.datetime.now(), "%m.%d %H:%M:%S")
    return now_str


class Timer(object):
    def __init__(self, name=None, quiet=False):
        self.name = name
        self.quiet = quiet

    def __enter__(self):
        if not self.quiet and self.name:
            print(get_now_str() + " : " + self.name + "...")

        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        elapsed = time.time() - self.tstart

        if not self.quiet:
            if self.name:
                print(
                    get_now_str() + " : " + self.name + " totaltime: {}".format(elapsed)
                )
            else:
                print(get_now_str() + " : " + "totaltime: {}".format(elapsed))
        # / if not quiet:
