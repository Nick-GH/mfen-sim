import os
import sys
from optparse import OptionParser

sys.path.insert(0, os.path.join(sys.path[0], ".."))
from feature_extractor.idascript import IDAScript
from feature_extractor.utils import do_multiprocess

if __name__ == "__main__":
    op = OptionParser()
    op.add_option(
        "--indir", action="store", type=str, dest="indir", help="Input directory"
    )
    op.add_option(
        "--idapath",
        action="store",
        type=str,
        dest="idapath",
        default="",
        help="IDA directory path",
    )
    op.add_option(
        "--idc",
        action="store",
        type=str,
        dest="idc",
        default="feature_extractor/ida/fetch_funcdata.py",
        help="IDA script file",
    )
    op.add_option(
        "--idcargs",
        action="store",
        type=str,
        dest="idcargs",
        default="",
        help="arguments seperated by ',' (e.g. --idcargs a,b,c,d)",
    )
    op.add_option("--force", action="store_true", dest="force")
    op.add_option("--log", action="store_true", dest="log")
    op.add_option("--stdout", action="store_true", dest="stdout")
    op.add_option("--debug", action="store_true", dest="debug")
    op.add_option(
        "--input_list",
        action="store",
        type=str,
        dest="input_list",
        help="A file containing paths of target binaries",
    )

    (opts, args) = op.parse_args()
    assert opts.input_list and os.path.exists(opts.input_list)

    idascript = IDAScript(
        idapath=opts.idapath,
        idc=opts.idc,
        idcargs=opts.idcargs,
        force=opts.force,
        log=opts.log,
        stdout=opts.stdout,
        debug=opts.debug,
    )
    idascript.run(opts.input_list)
