from cicflowmeter import sniffer
from scapy.sendrecv import AsyncSniffer as _AsyncSniffer

_original_create_sniffer = sniffer.create_sniffer


class _OfflineSniffer(_AsyncSniffer):
    def __init__(self, *args, **kwargs):
        if kwargs.get("offline") is not None:
            kwargs["filter"] = None
        super().__init__(*args, **kwargs)


def _create_sniffer(input_file, input_interface, output_mode, output,
                    input_directory=None, fields=None, verbose=False):
    return _original_create_sniffer(
        input_file,
        input_interface,
        output_mode,
        output,
        fields=input_directory,
        verbose=fields,
    )


sniffer.create_sniffer = _create_sniffer
sniffer.AsyncSniffer = _OfflineSniffer
sniffer.main()
