"""``python -m fastmd.distributed`` (run under torchrun): the command line lives in the entry-point package,
:mod:`fastmd.interfaces.distributed`."""
from fastmd.interfaces.distributed import main

if __name__ == "__main__":
    main()
